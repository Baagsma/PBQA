"""Shared in-memory HTTP fakes and DB stub for backend unit tests."""

import json as _json
from enum import Enum
from urllib.parse import urlsplit

import requests as real_requests
from pydantic import BaseModel

MODEL = "testmodel"
HOST = "fakehost"
PORT = 8080


# =============================================================================
# Schemas
# =============================================================================


class Weather(BaseModel):
    temperature: float
    condition: str


class Reply(BaseModel):
    reply: str


class Color(str, Enum):
    red = "red"
    blue = "blue"


class Item(BaseModel):
    color: Color
    note: str


# =============================================================================
# Fake servers
# =============================================================================


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeServer:
    """Mimics a llama.cpp server's HTTP surface, recording every call."""

    def __init__(self, rerank=False, total_slots=1):
        self.calls = []  # (method, path_with_query, body)
        self.rerank = rerank
        self.total_slots = total_slots
        self.chat_content = _json.dumps({"temperature": 20.0, "condition": "sunny"})
        self.chat_error = None  # exception to raise on /v1/chat/completions
        self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        self.rerank_results = []

    def handle(self, method, path, body):
        self.calls.append((method, path, body))

        if method == "GET" and path == "/health":
            return FakeResponse({"status": "ok"})
        if method == "GET" and path == "/props":
            return FakeResponse({"total_slots": self.total_slots})
        if method == "GET" and path == "/v1/models":
            # llama.cpp serves the OpenAI endpoint too, with its own owner
            return FakeResponse(
                {
                    "object": "list",
                    "data": [{"id": MODEL, "object": "model", "owned_by": "llamacpp"}],
                }
            )
        if path == "/v1/rerank":
            if self.rerank:
                return FakeResponse({"results": self.rerank_results})
            return FakeResponse({"error": {"code": 501, "message": "no rerank"}})
        if path.startswith("/slots/"):
            if "action=restore" in path and body["filename"].startswith(
                "zppivzcjfxvavwyqxse"
            ):
                # The capability probe uses a garbage filename; a 400 means
                # the endpoint exists (slot saving enabled on the server)
                return FakeResponse({"error": {"code": 400, "message": "not found"}})
            return FakeResponse({"id_slot": 0})
        if path == "/v1/chat/completions":
            if self.chat_error:
                raise self.chat_error
            return FakeResponse(
                {
                    "choices": [{"message": {"content": self.chat_content}}],
                    "usage": self.usage,
                }
            )

        raise AssertionError(f"Unexpected request: {method} {path}")

    def calls_to(self, path_prefix):
        return [c for c in self.calls if c[1].startswith(path_prefix)]


class FakeVLLMServer:
    """Mimics a vLLM OpenAI-compatible server, recording every call."""

    def __init__(self, model_id="qwen3.6-27b-nvfp4", strict_model=False):
        self.calls = []  # (method, path_with_query, body)
        self.model_id = model_id
        self.strict_model = strict_model  # reject requests for other model ids
        self.chat_content = _json.dumps({"temperature": 20.0, "condition": "sunny"})
        self.chat_error = None  # exception to raise on /v1/chat/completions
        self.error_payload = None  # OpenAI-style error object to return instead
        self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def handle(self, method, path, body):
        self.calls.append((method, path, body))

        if method == "GET" and path == "/health":
            return FakeResponse({}, status_code=200)
        if method == "GET" and path == "/props":
            # FastAPI 404 with a JSON body, as real vLLM answers it
            return FakeResponse({"detail": "Not Found"}, status_code=404)
        if method == "GET" and path == "/v1/models":
            return FakeResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": self.model_id, "object": "model", "owned_by": "vllm"}
                    ],
                }
            )
        if path == "/v1/chat/completions":
            if self.chat_error:
                raise self.chat_error
            if self.error_payload:
                return FakeResponse(self.error_payload)
            if self.strict_model and body.get("model") != self.model_id:
                # Real vLLM's 404 payload for an unknown model id
                return FakeResponse(
                    {
                        "object": "error",
                        "message": f"The model `{body.get('model')}` does not exist.",
                        "type": "NotFoundError",
                        "code": 404,
                    },
                    status_code=404,
                )
            return FakeResponse(
                {
                    "choices": [{"message": {"content": self.chat_content}}],
                    "usage": self.usage,
                }
            )

        raise AssertionError(f"Unexpected request: {method} {path}")

    def calls_to(self, path_prefix):
        return [c for c in self.calls if c[1].startswith(path_prefix)]


class FakeTransport:
    """Drop-in replacement for the requests module inside backend code."""

    exceptions = real_requests.exceptions

    def __init__(self):
        self.servers = {}  # "host:port" -> FakeServer | FakeVLLMServer

    def add_server(self, host, port, **kwargs):
        server = FakeServer(**kwargs)
        self.servers[f"{host}:{port}"] = server
        return server

    def add_vllm_server(self, host, port, **kwargs):
        server = FakeVLLMServer(**kwargs)
        self.servers[f"{host}:{port}"] = server
        return server

    def _dispatch(self, method, url, body):
        parts = urlsplit(url)
        server = self.servers.get(parts.netloc)
        if server is None:
            raise real_requests.exceptions.ConnectionError(f"no server at {url}")
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        return server.handle(method, path, body)

    def get(self, url, **kwargs):
        return self._dispatch("GET", url, None)

    def post(self, url, headers=None, data=None, json=None, **kwargs):
        body = json if json is not None else (
            _json.loads(data) if data is not None else None
        )
        return self._dispatch("POST", url, body)


# =============================================================================
# Stub DB
# =============================================================================


class StubDB:
    def __init__(self, schema=None, system_prompt="You report the weather.",
                 base_examples=None):
        self.schema = schema or Weather.model_json_schema()
        self.system_prompt = system_prompt
        self.base_examples = base_examples if base_examples is not None else [
            {
                "input": "how hot is it?",
                "response": {"temperature": 25.0, "condition": "clear"},
                "metadata": {},
            }
        ]

    def get_patterns(self):
        return ["weather"]

    def get_metadata(self, pattern):
        metadata = {"schema": self.schema}
        if self.system_prompt:
            metadata["system_prompt"] = self.system_prompt
        return metadata

    def where(self, **kwargs):
        return list(self.base_examples)

    def query(self, *args, **kwargs):
        return []
