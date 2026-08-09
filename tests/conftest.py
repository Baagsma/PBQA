import pytest

import PBQA.backends.llamacpp as llamacpp_module
import PBQA.backends.ninfer as ninfer_module
import PBQA.backends.vllm as vllm_module
from tests.mock_transport import FakeTransport


@pytest.fixture
def transport(monkeypatch):
    """Replace the HTTP transport in all backend modules with an in-memory fake."""
    transport = FakeTransport()
    monkeypatch.setattr(llamacpp_module, "requests", transport)
    monkeypatch.setattr(ninfer_module, "requests", transport)
    monkeypatch.setattr(vllm_module, "requests", transport)
    return transport
