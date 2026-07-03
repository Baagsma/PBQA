from PBQA.backends.base import Backend, BackendConfig
from PBQA.backends.llamacpp import LlamaCppBackend
from PBQA.backends.vllm import VLLMBackend

# Engine name -> Backend class, used by LLM.connect_model(engine=...).
# Order matters for detection: llama.cpp is probed first because its /props
# endpoint is unique to it, while /v1/models exists on every OpenAI-compatible
# server.
ENGINES = {
    "llamacpp": LlamaCppBackend,
    "vllm": VLLMBackend,
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
    "LlamaCppBackend",
    "VLLMBackend",
    "ENGINES",
    "detect_engine",
]
