"""llama.cpp server backend.

Talks to a llama-server instance over HTTP. Cache strategy: each pattern gets
a cache slot and a per-pattern file (`{pattern}-{model}.bin`) persisted via the
server's `--slot-save-path` mechanism — restored before and saved after every
generation, so interleaved patterns each keep their own KV prefix across
requests and server restarts.
"""

import json
import logging
from time import time
from typing import List

import requests

from PBQA.backends.base import (
    Backend,
    BackendConfig,
    EngineDriftError,
    error_body,
)

log = logging.getLogger("PBQA.backends.llamacpp")

DEFAULT_TOTAL_SLOTS = 1096
DETECT_TIMEOUT = 5


class LlamaCppBackend(Backend):
    warm_on_link = False  # slot files are durable; no prefill needed

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.total_slots = DEFAULT_TOTAL_SLOTS
        self._slots = {}  # pattern -> slot id

    @classmethod
    def detect(cls, config: BackendConfig) -> bool:
        # /props only exists on llama.cpp; other OpenAI-compatible servers
        # (vLLM among them) answer it with a 404
        try:
            response = requests.get(
                config.base_url + "/props", timeout=DETECT_TIMEOUT
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def connect(self) -> None:
        props = self.get_props()
        if props == {}:
            raise ValueError(
                f"Failed to connect to LLM server at {self.config.address}"
            )

        self.total_slots = props.get("total_slots", DEFAULT_TOTAL_SLOTS)
        self.is_rerank = self._probe_rerank()
        self.store_cache = (
            self.config.store_cache
            and not self.is_rerank
            and self._probe_store_cache()
        )
        self.connected = True

    def health(self) -> bool:
        try:
            requests.get(self.config.base_url + "/health")
            return True
        except requests.exceptions.RequestException:
            log.warning(
                f"Failed to connect to LLM server at {self.config.address}. Ensure the server is running and the host and port are correct."
            )
            return False

    def generate(
        self,
        messages: List[dict],
        schema: dict | None,
        pattern: str,
        model: str,
        overrides: dict,
        use_cache: bool = True,
    ) -> dict:
        slot = self._get_cache_slot(pattern)

        if self.store_cache:
            self._load_cache(pattern, model, slot)

        log.info(
            f"Performing query ({pattern}-{model}) at "
            f"{self.config.address} ID slot {slot}"
        )

        def send() -> dict:
            return self._chat_completion(
                messages, schema, model, slot, overrides, use_cache
            )

        then = time()
        response = self._recover_unsupported_params(send(), overrides, send)
        if "error" in response or response.get("object") == "error":
            # llama.cpp ignores the request's model name entirely - a server
            # that REJECTS it as unknown is not llama.cpp (vLLM's 404, in
            # either its nested {"error": {...}} or flat {"object": "error"}
            # shape). The engine behind this address has changed; retrying
            # in llama.cpp dialect can never succeed.
            err = error_body(response)
            message = err.get("message", "")
            err_type = err.get("type", "")
            if "does not exist" in message or err_type == "NotFoundError":
                raise EngineDriftError(
                    f"Server at {self.config.address} rejected the model name "
                    f"({message!r}) - llama.cpp never does this. The server "
                    f"was likely replaced by a different engine since connect."
                )
            raise ValueError(f"LLM error:\n{json.dumps(response, indent=4)}")

        choice = response["choices"][0]
        content = choice["message"]["content"]
        response_time = time() - then

        if self.store_cache:
            self._save_cache(pattern, model, slot)

        return {
            "content": content,
            "usage": response["usage"],
            "response_time": response_time,
            "finish_reason": choice.get("finish_reason"),
        }

    def _chat_completion(
        self,
        messages: List[dict],
        schema: dict | None,
        model: str,
        slot: int,
        overrides: dict,
        use_cache: bool,
    ) -> dict:
        data = {
            "model": model,
            "id_slot": slot,
            "cache_prompt": use_cache,
            "messages": messages,
            **({"json_schema": schema} if schema else {}),
            **self._merge_request(overrides),
        }
        return requests.post(
            self.config.base_url + "/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer no-key",
            },
            data=json.dumps(data),
        ).json()

    def rerank(self, query: str, documents: List[str]) -> List[dict]:
        response = requests.post(
            self.config.base_url + "/v1/rerank",
            json={"query": query, "documents": documents},
        ).json()
        return response["results"]

    def get_props(self) -> dict:
        try:
            response = requests.get(self.config.base_url + "/props")
            if response.status_code != 200:
                # A reachable server without /props is not llama.cpp; an
                # error body would otherwise pass for valid (non-empty) props
                return {}
            return response.json()
        except requests.exceptions.RequestException:
            raise ValueError(
                f"Failed to get properties from LLM server at {self.config.address}. Ensure the server is running and the host and port are correct."
            )

    def _probe_rerank(self) -> bool:
        try:
            response = requests.post(
                self.config.base_url + "/v1/rerank",
                json={"query": "test", "documents": ["test"]},
            )
            if response.status_code != 200:
                # A router 404ing with e.g. {"detail": "Not Found"} has no
                # "error" key and would otherwise pass for rerank support
                return False
            if "error" in response.json():
                return False
            log.info(f"Model at {self.config.address} supports reranking")
            return True
        except requests.exceptions.RequestException:
            log.info(f"Model at {self.config.address} does not support reranking")
            return False

    def _probe_store_cache(self) -> bool:
        try:
            response = requests.post(
                self.config.base_url + "/slots/0?action=restore",
                json={"filename": "zppivzcjfxvavwyqxse.bin"},
            ).json()
            if "error" in response:
                if response["error"]["code"] == 400:
                    log.info(
                        f"Connection to slot saving endpoint at {self.config.address} successful"
                    )
                    return True
                log.warning(
                    f"Failed to connect to slot saving endpoint at {self.config.address} with error {response['error']['code']}: {response['error']['message']}"
                )
            return False
        except requests.exceptions.RequestException:
            log.info(
                f"Failed to connect to slot saving endpoint at {self.config.address}. Disabling cache saving."
            )
            return False

    def _get_cache_slot(self, pattern: str) -> int:
        if pattern not in self._slots:
            # Find the lowest slot not yet assigned to another pattern
            for slot in range(self.total_slots):
                if slot not in self._slots.values():
                    self._slots[pattern] = slot
                    break
            else:
                # If no slots are available, use the last slot
                self._slots[pattern] = self.total_slots - 1

        return self._slots[pattern]

    def _load_cache(self, pattern: str, model: str, slot: int) -> None:
        try:
            response = requests.post(
                self.config.base_url + f"/slots/{slot}?action=restore",
                json={"filename": f"{pattern}-{model}.bin"},
            ).json()
            if "error" in response:
                if response["error"]["code"] == 400:
                    log.info(f"Cache for {pattern}-{model} not found")
                    return
                log.warning(
                    f"Failed to load cache for {pattern}-{model} to slot {slot} with error {response['error']['code']}: {response['error']['message']}"
                )
                return
            log.info(f"Loaded cache for {pattern}-{model} to slot {slot}")
        except requests.exceptions.RequestException:
            log.warning(f"Failed to load cache for {pattern}-{model} to slot {slot}")

    def _save_cache(self, pattern: str, model: str, slot: int) -> None:
        try:
            requests.post(
                self.config.base_url + f"/slots/{slot}?action=save",
                json={"filename": f"{pattern}-{model}.bin"},
            )
            log.info(f"Saved cache for {pattern}-{model} to slot {slot}")
        except requests.exceptions.RequestException:
            log.warning(f"Failed to save cache for {pattern}-{model} to slot {slot}")
