"""NInfer server backend.

Talks to a NInfer (https://github.com/Neroued/ninfer) server over its
OpenAI-compatible HTTP API. Cache strategy: none, client- or server-side —
NInfer keeps a single resident sequence with one restore checkpoint, placed
at the previous request's full prompt (through the assistant header). A new
user turn diverges at the role token, strictly before that checkpoint, so
pattern-rotation workloads can never hit it and warming cannot help
(warm_on_link is False; a warm request would just pay the prefix prefill
without moving the checkpoint anywhere useful). Reuse only occurs for exact
re-runs and token-exact conversation continuations. Prefill is fast enough
(~10k tok/s on the target card) that cold pattern prefixes cost ~100ms per
1k tokens.

Structured output: NInfer has no grammar-constrained decoding and rejects any
response_format other than {type: text}. Schemas are instead rendered into the
system message (inside the fixed, cacheable prefix) and the completion is
validated client-side against the schema, with one jittered retry on
violation. This is soft enforcement — the pattern's few-shot examples carry
most of the weight in practice — so treat outputs as validated, not
guaranteed-by-construction like the llama.cpp/vLLM backends.

Quirks vs the other OpenAI-dialect engines: the request model field must equal
the server's --model-id exactly (discovered from /v1/models, owned_by
"ninfer"); min_p and other unknown sampling fields are accepted and silently
ignored; top_k is kernel-capped at 20; reasoning arrives separately as
reasoning_content and never mixes into content.
"""

import json
import logging
from time import time
from typing import List

import jsonschema
import requests

from PBQA.backends.base import Backend, BackendConfig, error_body

log = logging.getLogger("PBQA.backends.ninfer")

DETECT_TIMEOUT = 5

SCHEMA_INSTRUCTION = (
    "Respond with a single JSON object (no code fences, no commentary) that "
    "validates against this JSON Schema:\n{schema}"
)


class NinferBackend(Backend):
    # NInfer's single-slot boundary checkpoint cannot serve a pattern prefix
    # (see module docstring); warming would burn a prefill for nothing
    warm_on_link = False

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.model_id = None  # served model alias, discovered on connect

    @classmethod
    def detect(cls, config: BackendConfig) -> bool:
        # NInfer stamps owned_by="ninfer" on /v1/models — an exact
        # discriminator, like vLLM's
        try:
            response = requests.get(
                config.base_url + "/v1/models", timeout=DETECT_TIMEOUT
            )
            if response.status_code != 200:
                return False
            data = response.json().get("data", [])
            return bool(data) and data[0].get("owned_by") == "ninfer"
        except (requests.exceptions.RequestException, ValueError):
            return False

    def connect(self) -> None:
        try:
            response = requests.get(self.config.base_url + "/health")
        except requests.exceptions.RequestException:
            raise ValueError(
                f"Failed to connect to NInfer server at {self.config.address}. Ensure the server is running and the host and port are correct."
            )
        if response.status_code != 200:
            raise ValueError(
                f"NInfer server at {self.config.address} is unhealthy (status {response.status_code})"
            )

        # Unlike vLLM there is no startup window: ninfer-serve binds HTTP only
        # after the artifact is resident, so the alias is available immediately
        self._discover_model_id()
        if self.model_id is None:
            raise ValueError(
                f"NInfer server at {self.config.address} lists no model"
            )

        if self.config.store_cache:
            log.info(
                f"NInfer at {self.config.address} manages caching automatically "
                f"(compatible-prefix reuse); no client-side cache persistence."
            )
        self.store_cache = False
        self.is_rerank = False
        self.connected = True

    def _discover_model_id(self) -> None:
        """Read the served model alias from /v1/models.

        Called on connect and again when the server reports model_not_found —
        which happens when a server with a different --model-id was deployed
        on the same port after this backend connected.
        """
        try:
            models = requests.get(self.config.base_url + "/v1/models").json()
        except requests.exceptions.RequestException:
            raise ValueError(
                f"Failed to list models on NInfer server at {self.config.address}"
            )
        data = models.get("data", [])
        if data:
            self.model_id = data[0]["id"]
            log.info(
                f"NInfer server at {self.config.address} serves model {self.model_id}"
            )

    def health(self) -> bool:
        try:
            response = requests.get(self.config.base_url + "/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            log.warning(
                f"Failed to connect to NInfer server at {self.config.address}. Ensure the server is running and the host and port are correct."
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
                "use_cache=False is ignored on NInfer; prefix reuse is automatic"
            )

        log.info(
            f"Performing query ({pattern}-{model}) at {self.config.address}"
        )

        if schema:
            messages = self._inject_schema(messages, schema)

        def send(extra: dict = {}) -> dict:
            return self._chat_completion(messages, model, {**overrides, **extra})

        then = time()
        response = send()
        if self._is_unknown_model_error(response):
            # A server with a different --model-id was deployed on this port
            # since we connected; rediscover the alias and retry once
            stale = self.model_id
            self._discover_model_id()
            if self.model_id != stale:
                log.warning(
                    f"Served model at {self.config.address} changed "
                    f"({stale} -> {self.model_id}); retrying"
                )
                response = send()
        response = self._recover_unsupported_params(
            response, overrides, send
        )
        if "error" in response or response.get("object") == "error":
            raise ValueError(f"LLM error:\n{json.dumps(response, indent=4)}")

        content = self._clean_content(
            response["choices"][0]["message"]["content"]
        )

        if schema:
            violation = self._schema_violation(content, schema)
            if violation:
                # Soft enforcement: the model missed the schema. One retry
                # with sampling jitter (same rationale as the malformed-JSON
                # retry in llm.py: identical params replay identical output)
                log.warning(
                    f"Response from {self.config.address} for {pattern} "
                    f"violates the schema: {violation}. Retrying once with "
                    f"sampling jitter."
                )
                response = send(
                    {
                        "temperature": max(
                            0.4, float(overrides.get("temperature") or 0)
                        ),
                        "presence_penalty": 1.0,
                    }
                )
                if "error" in response or response.get("object") == "error":
                    raise ValueError(
                        f"LLM error:\n{json.dumps(response, indent=4)}"
                    )
                content = self._clean_content(
                    response["choices"][0]["message"]["content"]
                )
                violation = self._schema_violation(content, schema)
                if violation:
                    raise ValueError(
                        f"Model returned schema-violating output for pattern "
                        f"'{pattern}' after retry: {violation}. "
                        f"content: {content[-300:]!r}"
                    )

        return {
            "content": content,
            "usage": response["usage"],
            "response_time": time() - then,
            "finish_reason": response["choices"][0].get("finish_reason"),
        }

    def _chat_completion(
        self,
        messages: List[dict],
        model: str,
        overrides: dict,
    ) -> dict:
        data = {
            # NInfer rejects any model other than its --model-id, so the
            # discovered alias is the only name that can ever succeed
            "model": self.model_id or model,
            "messages": messages,
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
    def _inject_schema(messages: List[dict], schema: dict) -> List[dict]:
        """Render the schema instruction into the system message.

        The system message sits at the head of the pattern's fixed prefix, and
        the schema is constant per pattern — so the instruction lands inside
        the cacheable region instead of invalidating it per request. The
        caller's list is not mutated.
        """
        instruction = SCHEMA_INSTRUCTION.format(schema=json.dumps(schema))
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n\n" + instruction,
            }
        else:
            messages.insert(0, {"role": "system", "content": instruction})
        return messages

    @staticmethod
    def _clean_content(content: str) -> str:
        """Strip markdown code fences from a completion.

        Grammarless models occasionally wrap the JSON in ```json fences
        despite the instruction; the fence is presentation, not content.
        """
        content = content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = content[content.index("\n") + 1 : -3].strip()
        return content

    @staticmethod
    def _schema_violation(content: str, schema: dict) -> str | None:
        """Why content violates schema, or None if it validates.

        Parse failures are left for llm.py's malformed-JSON handling — its
        retry already covers truncation and runaway loops; this check only
        covers well-formed JSON of the wrong shape, which grammar-backed
        engines exclude by construction and NInfer cannot.
        """
        try:
            instance = json.loads(content)
        except json.JSONDecodeError:
            return None
        try:
            jsonschema.validate(instance=instance, schema=schema)
        except jsonschema.ValidationError as e:
            return e.message
        except jsonschema.SchemaError as e:
            log.warning(f"Unvalidatable schema, skipping validation: {e}")
        return None

    @staticmethod
    def _is_unknown_model_error(response: dict) -> bool:
        error = error_body(response)
        if error is None:
            return False
        return error.get("code") == "model_not_found"

    # warm() stays the base no-op: NInfer's restore checkpoint sits past the
    # assistant header of whatever prompt was warmed, and every real query
    # diverges at its user turn before reaching it — a warm request can never
    # produce a hit (verified against ninfer 455c13c request logs, cache=0).
