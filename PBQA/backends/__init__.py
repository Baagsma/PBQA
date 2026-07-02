from PBQA.backends.base import Backend, BackendConfig
from PBQA.backends.llamacpp import LlamaCppBackend

# Engine name -> Backend class, used by LLM.connect_model(engine=...)
ENGINES = {
    "llamacpp": LlamaCppBackend,
}

__all__ = ["Backend", "BackendConfig", "LlamaCppBackend", "ENGINES"]
