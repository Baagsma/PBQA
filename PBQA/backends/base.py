"""Backend abstraction for inference servers.

A Backend represents a single running inference server (one host:port). It owns
everything engine-specific: capability probing, request payload assembly, and
the cache strategy (explicit slot save/restore for llama.cpp, automatic prefix
caching for vLLM). The LLM class sits above this interface and handles the
pattern layer, schema preprocessing, and routing.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List

log = logging.getLogger("PBQA.backends")

# A server refusing a sampling parameter names it in the refusal, e.g. vLLM
# with speculative decoding: "The min_p and logit_bias sampling parameters are
# not yet supported with speculative decoding." Anything else in the sentence
# is prose, filtered out by matching against the parameters actually sent.
UNSUPPORTED_SAMPLING_PARAMS = re.compile(
    r"(?P<params>[\w, ]+?)\s+sampling parameters?\s+(?:is|are)\s+not(?: yet)? supported",
    re.IGNORECASE,
)


def error_body(response: dict) -> dict | None:
    """The OpenAI-style error object from a response, or None if there is none.

    Servers use two shapes for the same error: nested ({"error": {...}}) and
    flat ({"object": "error", "message": ...}).
    """
    if not ("error" in response or response.get("object") == "error"):
        return None
    error = response.get("error", response)
    return error if isinstance(error, dict) else {"message": str(error)}


class EngineDriftError(ValueError):
    """The server behind this backend's address is no longer the engine this
    backend speaks.

    Raised when a response carries another engine's unmistakable signature —
    e.g. a llama.cpp-dialect request (alias model name) answered with vLLM's
    "model does not exist" 404. Happens when a different inference server is
    deployed on the same host:port after connect() (an engine swap behind a
    proxy, a llama.cpp box replaced by vLLM, ...). The caller should redetect
    the engine and rebuild the backend rather than retry as-is.
    """


@dataclass
class BackendConfig:
    """Configuration for a single inference server endpoint.

    Attributes:
    - host/port: Server address.
    - strict_schema: Set additionalProperties to false on all object types in
      JSON schemas. Required for servers using llguidance-based grammar
      enforcement, which defaults additionalProperties to true per the JSON
      Schema spec.
    - store_cache: Whether to persist per-pattern KV caches (engines that
      support it). The effective value is decided during connect() based on
      server capabilities.
    - request_defaults: Default generation parameters (temperature, min_p,
      max_tokens, stop, ...) merged into every request payload. This is the
      only part of the config that ever reaches the wire.
    """

    host: str
    port: int
    strict_schema: bool = False
    store_cache: bool = True
    request_defaults: dict = field(default_factory=dict)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class Backend(ABC):
    """A single inference server endpoint.

    Lifecycle: construct with a BackendConfig, then connect() before use.
    Fallback backends may defer connect() until first request (lazy).
    """

    # Whether LLM.link() should prefill the pattern prefix on this backend.
    # False for engines with durable caches (llama.cpp slot files), True for
    # engines whose cache dies with the process (vLLM in-VRAM prefix cache).
    warm_on_link: bool = False

    def __init__(self, config: BackendConfig):
        self.config = config
        self.connected = False
        # Capabilities, populated by connect()
        self.is_rerank = False
        self.store_cache = False
        # Sampling parameters this server refused, learned from its own error
        # message and kept out of every subsequent payload
        self._suppressed_params: set[str] = set()

    @classmethod
    def detect(cls, config: BackendConfig) -> bool:
        """Return True if the server at config looks like this engine.

        Used to resolve engine="auto". Must not raise and should be a single
        cheap HTTP probe against an engine-specific endpoint.
        """
        return False

    @abstractmethod
    def connect(self) -> None:
        """Probe the server and populate capabilities.

        Raises ValueError if the server is unreachable or unusable.
        """

    @abstractmethod
    def health(self) -> bool:
        """Cheap liveness check. Must not raise."""

    @abstractmethod
    def generate(
        self,
        messages: List[dict],
        schema: dict | None,
        pattern: str,
        model: str,
        overrides: dict,
        use_cache: bool = True,
    ) -> dict:
        """Request one completion from the server.

        The payload is built from config.request_defaults merged with
        overrides, plus engine-specific fields — internal configuration never
        reaches the wire. `pattern` and `model` identify the KV cache for
        engines with explicit cache management.

        Returns a dict with:
        - "content" (str): The raw completion text.
        - "usage" (dict): Token usage as reported by the server.
        - "response_time" (float): Seconds spent on the completion request
          itself, excluding any cache save/restore overhead.

        Raises requests.exceptions.RequestException on transport failures
        (triggers failover) and ValueError on server-reported errors (aborts).
        """

    def warm(self, messages: List[dict], pattern: str, model: str) -> None:
        """Prefill the pattern prefix so subsequent requests hit the cache.

        Default no-op; engines with automatic prefix caching implement this as
        a minimal (max_tokens=1) request. Failures should be logged, not
        raised — warming is an optimization, not a correctness requirement.
        """

    def rerank(self, query: str, documents: List[str]) -> List[dict]:
        """Return the server's raw rerank results for the given documents.

        Only meaningful when is_rerank is True.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support reranking"
        )

    def _merge_request(self, overrides: dict) -> dict:
        """Merge request defaults with per-call overrides.

        Stop sequences are additive (defaults + per-call) rather than
        overriding, so per-call stop strings extend the connect-time ones.
        Parameters this server has refused never reach the wire again.
        """
        defaults = self.config.request_defaults
        merged = {**defaults, **overrides}
        merged["stop"] = list(defaults.get("stop", [])) + list(
            overrides.get("stop", [])
        )
        for param in self._suppressed_params:
            merged.pop(param, None)
        return merged

    def _recover_unsupported_params(
        self,
        response: dict,
        overrides: dict,
        send: Callable[[], dict],
    ) -> dict:
        """Retry once without the sampling parameters the server just refused.

        Some deployments reject otherwise valid requests over a sampling
        parameter they cannot serve — vLLM running speculative decoding 400s
        on min_p and logit_bias, and PBQA sends min_p by default. The refusal
        names the offending parameters, so they are dropped from this
        backend's payloads from here on: the failed round-trip is paid once
        per server, not once per request.

        Any other error is returned untouched for the caller to raise on.
        """
        refused = self._refused_params(response, overrides)
        if not refused:
            return response

        self._suppressed_params |= refused
        log.warning(
            f"Server at {self.config.address} does not support the "
            f"{', '.join(sorted(refused))} sampling parameter(s); dropping "
            f"them from every request to this backend and retrying."
        )
        return send()

    def _refused_params(self, response: dict, overrides: dict) -> set[str]:
        """Sampling parameters a response names as unsupported.

        Only names this backend actually sends count — the rest of the
        sentence is prose. Names already suppressed are excluded, so a server
        that keeps failing after the strip raises instead of retrying again.
        """
        error = error_body(response)
        if error is None:
            return set()

        match = UNSUPPORTED_SAMPLING_PARAMS.search(str(error.get("message", "")))
        if not match:
            return set()

        named = set(re.findall(r"[a-z_][a-z0-9_]*", match.group("params").lower()))
        sent = set(self.config.request_defaults) | set(overrides)
        return (named & sent) - self._suppressed_params
