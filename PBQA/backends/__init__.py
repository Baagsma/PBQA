from PBQA.backends.base import Backend, BackendConfig
from PBQA.backends.llamacpp import LlamaCppBackend
from PBQA.backends.vllm import VLLMBackend

# Engine name -> Backend class, used by LLM.connect_model(engine=...)
ENGINES = {
    "llamacpp": LlamaCppBackend,
    "vllm": VLLMBackend,
}

__all__ = ["Backend", "BackendConfig", "LlamaCppBackend", "VLLMBackend", "ENGINES"]
