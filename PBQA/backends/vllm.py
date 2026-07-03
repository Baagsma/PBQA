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
from time import time
from typing import List

import requests

from PBQA.backends.base import Backend, BackendConfig

log = logging.getLogger("PBQA.backends.vllm")


class VLLMBackend(Backend):
    warm_on_link = True  # prefix cache is VRAM-only; prefill after (re)starts

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.model_id = None  # served model name, discovered on connect

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

        if self.config.store_cache:
            log.info(
                f"vLLM at {self.config.address} manages caching automatically "
                f"(prefix caching); no client-side cache persistence."
            )
        self.store_cache = False
        self.is_rerank = False
        self.connected = True

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

        data = {
            "model": self.model_id or model,
            "messages": messages,
            **({"structured_outputs": {"json": schema}} if schema else {}),
            **self._merge_request(overrides),
        }

        log.info(
            f"Performing query ({pattern}-{model}) at {self.config.address}"
        )

        then = time()
        response = requests.post(
            self.config.base_url + "/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer no-key",
            },
            data=json.dumps(data),
        ).json()
        if "error" in response or response.get("object") == "error":
            raise ValueError(f"LLM error:\n{json.dumps(response, indent=4)}")

        return {
            "content": response["choices"][0]["message"]["content"],
            "usage": response["usage"],
            "response_time": time() - then,
        }

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
