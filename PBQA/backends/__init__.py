from PBQA.backends.base import Backend, BackendConfig, EngineDriftError
from PBQA.backends.llamacpp import LlamaCppBackend
from PBQA.backends.vllm import VLLMBackend

# Engine name -> Backend class, used by LLM.connect_model(engine=...).
# Order matters for detection: vLLM is probed first because its discriminator
# is exact (owned_by == "vllm" on /v1/models) and a real llama.cpp server can
# never match it. llama.cpp's /props probe goes last — a router can answer
# /props on behalf of a different backend than the one serving completions,
# which misdetects vLLM-served models as llama.cpp (and their schemas then
# get sent in a field vLLM silently ignores).
ENGINES = {
    "vllm": VLLMBackend,
    "llamacpp": LlamaCppBackend,
}


def detect_engine(config: BackendConfig) -> str:
    """Probe the server at config and return the name of the matching engine.

    Raises ValueError if no registered engine recognizes the server (which
    includes the server simply being unreachable).
    """
    for name, cls in ENGINES.items():
        if cls.detect(config):
            return name
    raise ValueError(
        f"Could not detect the inference engine at {config.address}. "
        f"Ensure the server is running, or pass engine explicitly "
        f"(one of {list(ENGINES)})."
    )


__all__ = [
    "Backend",
    "BackendConfig",
    "EngineDriftError",
    "LlamaCppBackend",
    "VLLMBackend",
    "ENGINES",
    "detect_engine",
]
