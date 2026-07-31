"""DB(device=...) places the encoder on the requested torch device.

No servers: the Qdrant client is replaced by an in-memory fake, and the encoder
is pinned to the local model cache, so nothing here reaches the network.
"""

from types import SimpleNamespace

import pytest

import PBQA.db as db_module
from PBQA import DB


class FakeQdrantClient:
    """The slice of the Qdrant client DB.__init__ touches."""

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.created = []

    def get_collections(self):
        return SimpleNamespace(collections=[])

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        return True


class FakeEncoder:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr(db_module, "QdrantClient", FakeQdrantClient)


def test_device_forwarded_to_encoder(fake_client, monkeypatch):
    monkeypatch.setattr(db_module, "SentenceTransformer", FakeEncoder)

    db = DB(path="unused", device="cpu")

    assert db.encoder.kwargs["device"] == "cpu"


def test_device_defaults_to_none(fake_client, monkeypatch):
    """No device means SentenceTransformer picks, as it did before the parameter existed."""
    monkeypatch.setattr(db_module, "SentenceTransformer", FakeEncoder)

    db = DB(path="unused")

    assert db.encoder.kwargs["device"] is None


def test_encoder_runs_on_cpu(fake_client, monkeypatch):
    """The real encoder lands on the CPU when asked, GPU present or not."""
    from sentence_transformers import SentenceTransformer

    def cached_encoder(model_name, **kwargs):
        return SentenceTransformer(model_name, local_files_only=True, **kwargs)

    monkeypatch.setattr(db_module, "SentenceTransformer", cached_encoder)

    try:
        db = DB(path="unused", device="cpu")
    except Exception as e:  # model not in the local cache
        pytest.skip(f"encoder model unavailable offline: {e}")

    assert db.encoder.device.type == "cpu"
    assert next(db.encoder.parameters()).device.type == "cpu"
