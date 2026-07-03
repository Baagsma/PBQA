"""Backend abstraction for inference servers.

A Backend represents a single running inference server (one host:port). It owns
everything engine-specific: capability probing, request payload assembly, and
the cache strategy (explicit slot save/restore for llama.cpp, automatic prefix
caching for vLLM). The LLM class sits above this interface and handles the
pattern layer, schema preprocessing, and routing.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

log = logging.getLogger("PBQA.backends")


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
        """
        defaults = self.config.request_defaults
        merged = {**defaults, **overrides}
        merged["stop"] = list(defaults.get("stop", [])) + list(
            overrides.get("stop", [])
        )
        return merged
