"""vLLM server backend.

Talks to a vLLM OpenAI-compatible server over HTTP. Cache strategy: none
client-side — vLLM's automatic prefix caching is a block-level hash table over
the whole KV pool keyed on token content, so a pattern's fixed example prefix
hits regardless of how patterns are interleaved. The cache lives in VRAM and
dies with the server process, hence warm_on_link: LLM.link() prefills each
pattern's fixed prefix so first queries after a server start hit the cache.

Structured output uses the `structured_outputs` request field (vLLM >= 0.12;
the older guided_json field was removed in v0.12.0).
"""

import json
import logging
from time import sleep, time
from typing import List

import requests

from PBQA.backends.base import Backend, BackendConfig, error_body

log = logging.getLogger("PBQA.backends.vllm")

DETECT_TIMEOUT = 5

# vLLM binds HTTP before the engine finishes loading; /v1/models answers 200
# with an empty list until the model is registered (minutes for large models).
# connect() waits this long for a model to appear before failing loud.
MODEL_LOAD_TIMEOUT = 300
MODEL_LOAD_POLL = 2.0


class VLLMBackend(Backend):
    warm_on_link = True  # prefix cache is VRAM-only; prefill after (re)starts

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.model_id = None  # served model name, discovered on connect

    @classmethod
    def detect(cls, config: BackendConfig) -> bool:
        # vLLM stamps owned_by="vllm" on /v1/models; llama.cpp serves the
        # endpoint too, so the owner is the discriminator, not the status
        try:
            response = requests.get(
                config.base_url + "/v1/models", timeout=DETECT_TIMEOUT
            )
            if response.status_code != 200:
                return False
            data = response.json().get("data", [])
            return bool(data) and data[0].get("owned_by") == "vllm"
        except (requests.exceptions.RequestException, ValueError):
            return False

    def connect(self) -> None:
        try:
            response = requests.get(self.config.base_url + "/health")
        except requests.exceptions.RequestException:
            raise ValueError(
                f"Failed to connect to vLLM server at {self.config.address}. Ensure the server is running and the host and port are correct."
            )
        if response.status_code != 200:
            raise ValueError(
                f"vLLM server at {self.config.address} is unhealthy (status {response.status_code})"
            )

        self._discover_model_id()

        # The server being up is not the model being ready: during startup
        # /v1/models is an empty list. Reporting "connected" with no model id
        # would make every request fall back to the alias name, which vLLM
        # rejects - so wait for registration, loudly, and fail loudly.
        if self.model_id is None:
            deadline = time() + MODEL_LOAD_TIMEOUT
            polls = 0
            while self.model_id is None and time() < deadline:
                if polls % 15 == 0:
                    log.info(
                        f"vLLM server at {self.config.address} is up but no "
                        f"model is registered yet (still loading?); waiting"
                    )
                polls += 1
                sleep(MODEL_LOAD_POLL)
                self._discover_model_id()
            if self.model_id is None:
                raise ValueError(
                    f"vLLM server at {self.config.address} did not register "
                    f"a model within {MODEL_LOAD_TIMEOUT}s"
                )

        if self.config.store_cache:
            log.info(
                f"vLLM at {self.config.address} manages caching automatically "
                f"(prefix caching); no client-side cache persistence."
            )
        self.store_cache = False
        self.is_rerank = False
        self.connected = True

    def _discover_model_id(self) -> None:
        """Read the served model id from /v1/models.

        Called on connect and again when the server reports an unknown model —
        which happens when a different model was deployed on the same port
        after this backend connected.
        """
        try:
            models = requests.get(self.config.base_url + "/v1/models").json()
        except requests.exceptions.RequestException:
            raise ValueError(
                f"Failed to list models on vLLM server at {self.config.address}"
            )
        data = models.get("data", [])
        if data:
            self.model_id = data[0]["id"]
            log.info(
                f"vLLM server at {self.config.address} serves model {self.model_id}"
            )

    def health(self) -> bool:
        try:
            response = requests.get(self.config.base_url + "/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            log.warning(
                f"Failed to connect to vLLM server at {self.config.address}. Ensure the server is running and the host and port are correct."
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
        if not use_cache:
            log.debug(
                "use_cache=False is ignored on vLLM; prefix caching is automatic"
            )

        log.info(
            f"Performing query ({pattern}-{model}) at {self.config.address}"
        )

        # Never send the alias: vLLM only accepts its served id. A missing id
        # here means the server had no model registered when we last looked -
        # try once more, then refuse with the reason instead of a bare 404.
        if self.model_id is None:
            self._discover_model_id()
            if self.model_id is None:
                raise ValueError(
                    f"vLLM server at {self.config.address} has no model "
                    f"registered (still loading?); request for pattern "
                    f"'{pattern}' not sent - the alias '{model}' would "
                    f"never match a served model."
                )

        def send() -> dict:
            return self._chat_completion(messages, schema, model, overrides)

        then = time()
        response = send()
        if self._is_unknown_model_error(response):
            # A different model was deployed on this port since we connected;
            # rediscover the served id and retry once
            stale = self.model_id
            self._discover_model_id()
            if self.model_id != stale:
                log.warning(
                    f"Served model at {self.config.address} changed "
                    f"({stale} -> {self.model_id}); retrying"
                )
                response = send()
        response = self._recover_unsupported_params(response, overrides, send)
        if "error" in response or response.get("object") == "error":
            raise ValueError(f"LLM error:\n{json.dumps(response, indent=4)}")

        return {
            "content": response["choices"][0]["message"]["content"],
            "usage": response["usage"],
            "response_time": time() - then,
            "finish_reason": response["choices"][0].get("finish_reason"),
        }

    def _chat_completion(
        self,
        messages: List[dict],
        schema: dict | None,
        model: str,
        overrides: dict,
    ) -> dict:
        data = {
            "model": self.model_id or model,
            "messages": messages,
            **({"structured_outputs": {"json": schema}} if schema else {}),
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

    @staticmethod
    def _is_unknown_model_error(response: dict) -> bool:
        error = error_body(response)
        if error is None:
            return False
        return (
            "does not exist" in error.get("message", "")
            or error.get("type") == "NotFoundError"
        )

    def warm(self, messages: List[dict], pattern: str, model: str) -> None:
        # Chat templates typically require the conversation to end on a user
        # turn. The filler content sits past the shared prefix, so the cached
        # blocks still cover the system prompt and examples.
        messages = list(messages)
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": "."})

        data = {
            "model": self.model_id or model,
            "messages": messages,
            "max_tokens": 1,
        }

        try:
            response = requests.post(
                self.config.base_url + "/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer no-key",
                },
                data=json.dumps(data),
            ).json()
            if "error" in response or response.get("object") == "error":
                log.warning(
                    f"Failed to warm prefix cache for {pattern} at "
                    f"{self.config.address}: {json.dumps(response)}"
                )
                return
            log.info(
                f"Warmed prefix cache for {pattern} at {self.config.address} "
                f"({response.get('usage', {}).get('prompt_tokens', '?')} prompt tokens)"
            )
        except requests.exceptions.RequestException as e:
            log.warning(
                f"Failed to warm prefix cache for {pattern} at "
                f"{self.config.address}: {e}"
            )
