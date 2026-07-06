import json
import logging
import math
from time import time
from typing import List

import requests
from pydantic import BaseModel

from PBQA.backends import ENGINES, Backend, BackendConfig, detect_engine
from PBQA.db import DB, resolve_path, path_exists
from PBQA.schema import lock_schema, resolve_refs

log = logging.getLogger("PBQA.llm")


class LLM:
    DEFAULT_HIST_DURATION = 1000
    DEFAULT_USER_NAME = "user"
    DEFAULT_ASSISTANT_NAME = "assistant"
    DEFAULT_RESULT_COUNT = 50

    def __init__(
        self,
        db: DB,
        host: str = None,
    ):
        """
        Initialize the LLM (Language Learning Model.

        This function initializes the LLM with the specified model and database.

        Parameters:
        - db (DB): The database to use for storing and retrieving examples.
        - host (str): The host of the LLM server. Can also be passed when connecting model servers.
        """

        self.db = db
        self.host = host

        self.models: dict[str, Backend] = {}
        self.pattern_models = {}
        self._fallbacks: dict[str, list[Backend]] = {}
        self._health_cache = {}  # (host, port) -> (healthy: bool, expires_at: float)

        self.HEALTHY_TTL = 30    # seconds to cache healthy status
        self.UNHEALTHY_TTL = 5   # seconds to cache unhealthy status (shorter for faster recovery)

    def connect_model(
        self,
        model: str,
        port: int,
        host: str = None,
        temperature: float = 1.0,
        min_p: float = 0.02,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        stop: List[str] = [],
        store_cache: bool = True,
        strict_schema: bool = False,
        engine: str = "auto",
        **kwargs,
    ) -> Backend:
        """
        Connect to an LLM server.

        Parameters:
        - model (str): The model to use for generating responses.
        - port (int): The port of the LLM server.
        - host (str): The host of the LLM server.
        - temperature (float): The temperature to use for generating responses.
        - min_p (float): The minimum probability to use for generating responses.
        - top_p (float): The top probability to use for generating responses.
        - max_tokens (int): The maximum number of tokens to use for generating responses.
        - stop (List[str]): Strings to stop the response generation.
        - store_cache (bool): Whether to save the cache to disk.
        - strict_schema (bool): Whether to set additionalProperties to false on all
          object types in JSON schemas. Required for servers using llguidance-based
          grammar enforcement, which defaults additionalProperties to true per the
          JSON Schema spec.
        - engine (str): The inference engine serving the model. "auto"
          (default) probes the server and picks the matching registered
          backend; pass an explicit name ("llamacpp", "vllm", ...) to skip
          detection.
        - kwargs: Additional default parameters to pass when querying the LLM server.

        Returns:
        - Backend: The connected backend.
        """

        if not host:
            host = self.host
        if not host:
            raise ValueError("Failed to connect to LLM server. No host provided.")

        if engine != "auto" and engine not in ENGINES:
            raise ValueError(
                f'Unknown engine "{engine}". Available engines: '
                f'{["auto"] + list(ENGINES.keys())}'
            )

        config = BackendConfig(
            host=host,
            port=port,
            strict_schema=strict_schema,
            store_cache=store_cache,
            request_defaults={
                "temperature": temperature,
                "min_p": min_p,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stop": stop,
                **kwargs,
            },
        )

        if engine == "auto":
            engine = detect_engine(config)

        backend = ENGINES[engine](config)
        backend.connect()

        log.info(f'Connected to model "{model}" at {host}:{port} ({engine})')

        self.models[model] = backend

        return backend

    def add_fallback(
        self,
        model: str,
        host: str,
        port: int,
        lazy: bool = True,
        engine: str = None,
        **kwargs,
    ) -> None:
        """
        Register a fallback backend for an existing model.

        When the primary backend is unavailable, requests will automatically
        fall back to registered backends in priority order.

        Parameters:
        - model (str): Must match an existing model name from connect_model().
        - host (str): The host of the fallback server.
        - port (int): The port of the fallback server.
        - lazy (bool): If True, defer connection validation until first use.
        - engine (str): The inference engine of the fallback server. Defaults
          to the same engine as the primary backend; "auto" probes the server
          at registration time (the server must be reachable, even with lazy).
        - kwargs: Override temperature, max_tokens, etc. for this backend.
        """
        if model not in self.models:
            raise ValueError(
                f'Model "{model}" not found. Connect the primary model first '
                f"using connect_model() before adding fallbacks."
            )

        primary = self.models[model]

        strict_schema = kwargs.pop("strict_schema", primary.config.strict_schema)
        store_cache = kwargs.pop("store_cache", primary.config.store_cache)

        config = BackendConfig(
            host=host,
            port=port,
            strict_schema=strict_schema,
            store_cache=store_cache,
            request_defaults={**primary.config.request_defaults, **kwargs},
        )

        if engine is None:
            backend_cls = type(primary)
        elif engine == "auto":
            # Requires the fallback server to be reachable now, even when
            # lazy — probing is the whole point of "auto"
            backend_cls = ENGINES[detect_engine(config)]
        elif engine in ENGINES:
            backend_cls = ENGINES[engine]
        else:
            raise ValueError(
                f'Unknown engine "{engine}". Available engines: '
                f'{["auto"] + list(ENGINES.keys())}'
            )

        fallback = backend_cls(config)

        if not lazy:
            fallback.connect()

        if model not in self._fallbacks:
            self._fallbacks[model] = []
        self._fallbacks[model].append(fallback)

        log.info(
            f'Registered fallback for "{model}" at {host}:{port} '
            f"(lazy={lazy}, priority={len(self._fallbacks[model])})"
        )

    def _check_health(self, backend: Backend) -> bool:
        """Check if a backend is healthy, using TTL cache."""
        key = (backend.config.host, backend.config.port)
        cached = self._health_cache.get(key)
        if cached:
            healthy, expires_at = cached
            if time() < expires_at:
                return healthy

        healthy = backend.health()
        ttl = self.HEALTHY_TTL if healthy else self.UNHEALTHY_TTL
        self._health_cache[key] = (healthy, time() + ttl)
        return healthy

    def _mark_healthy(self, backend: Backend) -> None:
        """Mark a backend as healthy after a successful request."""
        key = (backend.config.host, backend.config.port)
        self._health_cache[key] = (True, time() + self.HEALTHY_TTL)

    def _mark_unhealthy(self, backend: Backend) -> None:
        """Mark a backend as unhealthy after a failed request."""
        key = (backend.config.host, backend.config.port)
        self._health_cache[key] = (False, time() + self.UNHEALTHY_TTL)

    def _get_backends(self, model: str) -> list[Backend]:
        """Get ordered list of backends: [primary, fallback1, fallback2, ...]."""
        backends = [self.models[model]]
        for fb in self._fallbacks.get(model, []):
            if not fb.connected:
                try:
                    fb.connect()
                    log.info(f"Lazy-connected fallback at {fb.config.address}")
                except Exception as e:
                    log.warning(
                        f"Failed to lazy-connect fallback at {fb.config.address}: {e}"
                    )
                    self._mark_unhealthy(fb)
                    continue
            backends.append(fb)
        return backends

    def _get_response(
        self,
        input: str | dict,
        pattern: str,
        model: str = None,
        system_prompt: str = None,
        history_name: str = None,
        include_system_prompt: bool = True,
        include_base_examples: bool = True,
        n_hist: int = 0,
        n_example: int = 0,
        min_d: float = None,
        use_cache: bool = True,
        schema: BaseModel = None,
        stop: List[str] = [],
        custom_history: List[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Get a response from the LLM server.

        Parameters:
        - input (str): The input to the LLM.
        - pattern (str): The pattern to use for generating the response.
        - model (str): The model to use for generating the response.
        - system_prompt (str): The system prompt to provide to the LLM.
        - history_name (str): The name of the history to use for generating the response.
        - include_system_prompt (bool): Whether to include the system message.
        - include_base_examples (bool): Whether to include the base examples.
        - n_hist (int): The number of historical examples to load from the database.
        - n_example (int): The number of examples to load from the database.
        - min_d (float): The minimum distance between the input and the examples.
        - use_cache (bool): Whether to use the cache for the response.
        - schema (BaseModel): The schema to use for the response.
        - stop (List[str]): Strings to stop the response generation.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - dict: The response from the LLM.
        """

        if pattern not in self.db.get_patterns():
            raise ValueError(
                f'Pattern "{pattern}" not found in patterns {self.db.get_patterns()}. Make sure to load the pattern first using the `db.load_pattern()` method.'
            )

        if not model:
            model = self.pattern_models.get(pattern, None)
            if not model:
                raise ValueError(
                    f'No model provided and no model assigned for pattern "{pattern}". Make sure to call `llm.link()` or provide a model when calling `llm.ask()`.'
                )
            log.info(
                f'No model provided. Using stored model "{model}" for pattern "{pattern}" as assigned by the last call to `llm.link()`.'
            )
        if model not in self.models:
            raise ValueError(
                f'Model "{model}" not found in models {self.models.keys()}. Make sure to connect the model first using the `llm.connect_model()` method.'
            )

        if self.models[model].is_rerank:
            raise ValueError(
                f'Model "{model}" is a reranking model. Make sure to use the `llm.rerank()` method instead of `llm.ask()`.'
            )

        log.info(f"Generating response from LLM")

        messages = self._format_messages(
            input=input,
            pattern=pattern,
            system_prompt=system_prompt,
            history_name=history_name,
            include_base_examples=include_base_examples,
            include_system_prompt=include_system_prompt,
            n_hist=n_hist,
            n_example=n_example,
            min_d=min_d,
            custom_history=custom_history,
            **kwargs,
        )

        metadata = self.db.get_metadata(pattern)

        # If the schema consists of a single str component, pass None instead of the schema
        # Exception: If the string has enum constraints (Literal types), keep the schema
        # Note: Optional types use "anyOf" instead of direct "type", so we use .get()
        prop_name = None
        schema = schema or metadata["schema"]
        if (
            len(schema["properties"]) == 1
            and (prop_name := list(schema["properties"].keys())[0])
            and schema["properties"][prop_name].get("type") == "string"
            and "enum" not in schema["properties"][prop_name]
        ):
            log.info(
                f"Schema consists of a single string component ({prop_name}). Passing None instead of the schema."
            )
            schema = None

        if schema:
            schema = resolve_refs(schema)

        overrides = {**kwargs, "stop": stop}

        last_error = None
        for backend in self._get_backends(model):
            if not self._check_health(backend):
                log.info(
                    f"Skipping unhealthy backend {backend.config.address} "
                    f"for {pattern}"
                )
                continue

            send_schema = (
                lock_schema(schema)
                if schema and backend.config.strict_schema
                else schema
            )

            try:
                result = backend.generate(
                    messages=messages,
                    schema=send_schema,
                    pattern=pattern,
                    model=model,
                    overrides=overrides,
                    use_cache=use_cache,
                )
                content = result["content"]
                try:
                    llm_response = json.loads(content) if schema else content
                except json.JSONDecodeError as decode_err:
                    # Grammar-constrained output can still arrive malformed —
                    # typically a generation cut at max_tokens mid-escape
                    # (finish_reason=length ⇒ runaway). Log the evidence,
                    # retry once uncached, then fail loudly with the tail.
                    log.warning(
                        f"Malformed JSON from {backend.config.address} for "
                        f"{pattern}: {decode_err}. "
                        f"finish_reason={result.get('finish_reason')}, "
                        f"completion_tokens={result.get('usage', {}).get('completion_tokens')}, "
                        f"len={len(content)}, tail={content[-200:]!r}. "
                        f"Retrying once without cache."
                    )
                    result = backend.generate(
                        messages=messages,
                        schema=send_schema,
                        pattern=pattern,
                        model=model,
                        overrides=overrides,
                        use_cache=False,
                    )
                    content = result["content"]
                    try:
                        llm_response = json.loads(content)
                    except json.JSONDecodeError as retry_err:
                        raise ValueError(
                            f"Model returned malformed JSON for pattern "
                            f"'{pattern}' after retry: {retry_err}. "
                            f"finish_reason={result.get('finish_reason')}, "
                            f"tail: {content[-300:]!r}"
                        ) from retry_err
                log.info(f"Response:\n{json.dumps(llm_response, indent=4)}")

                self._mark_healthy(backend)

                return {
                    "input": input,
                    "response": llm_response if schema else {prop_name: llm_response},
                    "metadata": {
                        "total_time": result["response_time"],
                        **result["usage"],
                    },
                }
            except requests.exceptions.RequestException as e:
                self._mark_unhealthy(backend)
                log.warning(
                    f"Backend {backend.config.address} failed for "
                    f"{pattern}, trying next: {e}"
                )
                last_error = e
                continue

        raise ValueError(
            f"All backends failed for model '{model}': {last_error}\n\n"
            f"Ensure at least one inference server is running."
        )

    def _format_messages(
        self,
        pattern: str,
        input: str | dict = None,
        system_prompt: str = None,
        history_name: str = None,
        include_base_examples: bool = True,
        include_system_prompt: bool = True,
        n_example: int = 0,
        n_hist: int = 0,
        hist_duration: int = DEFAULT_HIST_DURATION,
        min_d: float = None,
        user_name: str = DEFAULT_USER_NAME,
        assistant_name: str = DEFAULT_ASSISTANT_NAME,
        log_input: bool = False,
        log_messages: bool = False,
        custom_history: List[dict] = None,
        **kwargs,
    ) -> list[dict[str, str]]:
        """
        Format the messages for the LLM.

        Parameters:
        - pattern (str): The pattern to use for generating the response.
        - input (str): The input to the LLM.
        - system_prompt (str): The system prompt to provide to the LLM.
        - history_name (str): The name of the history to use for generating the response.
        - include_base_examples (bool): Whether to include the base messages.
        - include_system_prompt (bool): Whether to include the system message.
        - n_example (int): The number of examples to load from the database.
        - n_hist (int): The number of historical examples to load from the database.
        - hist_duration (int): The duration of the historical examples to load.
        - min_d (float): The minimum distance between the input and the examples.
        - user_name (str): The name of the user.
        - assistant_name (str): The name of the assistant.
        - custom_history (List[dict]): Custom conversation history to use instead of database retrieval.
          When provided, n_hist is ignored and database history lookup is bypassed.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - list[dict[str, str]]: The formatted messages.
        """

        metadata = self.db.get_metadata(pattern)

        def format(
            docs: list[dict],
            user: str = user_name,
            assistant: str = assistant_name,
        ) -> list[dict[str, str]]:
            if not docs:
                return []

            pattern_collection_docs = list(docs[0].keys()) == [
                "input",
                "response",
                "metadata",
            ]
            has_schema = "schema" in metadata

            messages = []
            if pattern_collection_docs:
                for doc in docs:
                    messages.append(
                        format_role(
                            user,
                            doc["input"],
                        )
                    )
                    messages.append(
                        format_role(
                            assistant,
                            doc["response"],
                        )
                    )
            elif has_schema:  # If the docs are from a non-pattern collection
                response_props = list(metadata["schema"]["properties"].keys())
                if not all(
                    prop in doc.keys() for prop in response_props for doc in docs
                ):
                    raise ValueError(
                        f"Response properties {response_props} not found in docs {[list(doc.keys()) for doc in docs]}"
                    )

                input_props = metadata["doc_input_properties"]

                for doc in docs:
                    messages.append(
                        format_role(
                            user,
                            {k: v for k, v in doc.items() if k in input_props},
                        )
                    )
                    messages.append(
                        format_role(
                            assistant,
                            {k: v for k, v in doc.items() if k in response_props},
                        )
                    )
            else:
                raise ValueError(
                    f"Pattern {pattern} has no schema or metadata. Cannot format messages."
                )

            return messages

        def format_role(
            role: str,
            doc: str | dict,
        ) -> dict[str, str]:
            if type(doc) == str:
                return {"role": role, "content": doc}
            if not type(doc) == dict:
                raise ValueError(
                    f"Invalid document type {type(doc)}. Expected dict or str, got {doc}"
                )
            if (
                len(doc) == 1 and type(doc[list(doc.keys())[0]]) == str
            ):  # If the only has a single component of type str, pass it as a string instead of a dict (in line with get_response)
                return {"role": role, "content": doc[list(doc.keys())[0]]}
            else:
                return {
                    "role": role,
                    "content": json.dumps(doc),
                }

        messages = []
        system_prompt = system_prompt or metadata.get("system_prompt", None)
        if include_system_prompt and system_prompt:
            messages += [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]

        base_examples = []
        if include_base_examples:
            base_examples = self.db.where(
                collection_name=pattern,
                n=999,
                base_example=True,
            )
            base_examples.reverse()
            messages += format(base_examples)

        log.info(f"Base examples: {len(base_examples)}")

        examples = []
        if input:
            query_input = input
            if type(input) == dict:
                input_key = metadata.get("input_key", "input")
                if not path_exists(input, input_key):
                    raise ValueError(
                        f"Input dict must contain {input_key} key/path, got {input.keys()}"
                    )
                query_input = resolve_path(input, input_key)

            examples = self.db.query(
                pattern,
                query_input,
                n=n_example,
                min_d=min_d,
                base_example={"ne": True},
                **kwargs,
            )
            messages += format(examples)

        log.info(f"Examples: {len(examples)}/{n_example}")

        hist = []
        if custom_history is not None:
            # Use custom history directly, bypass database
            log.info(f"Using custom history: {len(custom_history)} entries")
            hist = custom_history
            messages += format(hist)
        elif n_hist:
            hist = self.db.where(
                collection_name=history_name or pattern,
                start=time() - hist_duration,
                end=time(),
                n=n_hist,
                base_example={"ne": True},
            )  # TODO: If len(hist) == n_hist, remove n oldest responses - something something modulo

            # Assert that the history is sorted by time_added
            hist.reverse()
            if hist and "metadata" in hist[0]:
                assert hist == sorted(
                    hist, key=lambda x: x["metadata"]["time_added"]
                ), f"Expected the history to be sorted by time_added, got {json.dumps(hist, indent=4)}"
            else:
                assert hist == sorted(
                    hist, key=lambda x: x["time_added"]
                ), f"Expected the history to be sorted by time_added, got {json.dumps(hist, indent=4)}"

            messages += format(hist)

        log.info(f"History: {len(hist)}/{n_hist if custom_history is None else 'custom'}")

        if input is not None:
            messages.append(format_role(user_name, input))

        if log_input:
            log.info(f"Input:\n{json.dumps(messages[-1], indent=4)}")
        if log_messages:
            log.info(
                "Messages:\n"
                + "".join(
                    [
                        f'{message["role"][:6]}:\t{message["content"]}\n'
                        for message in messages
                    ]
                )
            )

        return messages

    def link(
        self,
        pattern: str,
        model: str,
        cache_slot: int = None,
    ) -> None:
        """
        Link a pattern to a model.

        Patterns linked to a model can be queried with `llm.ask()` without
        specifying the model each time.

        Parameters:
        - pattern (str): The pattern to link.
        - model (str): The model to link the pattern to.
        - cache_slot (int): Deprecated and ignored. Cache slots are managed
          internally by the backend.
        """

        if cache_slot is not None:
            log.warning(
                "The cache_slot parameter is deprecated and ignored. Cache "
                "slots are managed internally by the backend."
            )

        if pattern not in self.db.get_patterns():
            raise ValueError(
                f'Pattern "{pattern}" not found. Make sure to load the pattern first using the `db.load_pattern()` method.'
            )
        if model not in self.models:
            raise ValueError(
                f'Model "{model}" not found. Make sure to connect the model first using the `llm.connect_model()` method.'
            )

        self.pattern_models[pattern] = model

        log.info(f'Linked pattern "{pattern}" to model "{model}"')

        if self.models[model].warm_on_link:
            self.warm(pattern, model)

    def warm(
        self,
        pattern: str,
        model: str = None,
    ) -> None:
        """
        Prefill a pattern's fixed prefix (system prompt + base examples) so
        subsequent queries hit the backend's prefix cache.

        Only meaningful for engines whose cache does not survive a server
        restart (e.g. vLLM's in-VRAM prefix cache); a no-op for llama.cpp,
        whose slot caches are persisted to disk. Called automatically by
        `link()` when the backend requests it. Warming failures are logged,
        never raised.

        Parameters:
        - pattern (str): The pattern whose prefix to warm.
        - model (str): The model to warm the prefix on. Defaults to the model
          linked to the pattern.
        """

        if not model:
            model = self.pattern_models.get(pattern, None)
            if not model:
                raise ValueError(
                    f'No model provided and no model assigned for pattern "{pattern}". Make sure to call `llm.link()` or provide a model when calling `llm.warm()`.'
                )
        if model not in self.models:
            raise ValueError(
                f'Model "{model}" not found. Make sure to connect the model first using the `llm.connect_model()` method.'
            )

        backend = self.models[model]

        messages = self._format_messages(pattern=pattern, input=None)
        if not messages:
            log.info(
                f'Nothing to warm for pattern "{pattern}" (no system prompt or base examples)'
            )
            return

        try:
            backend.warm(messages, pattern, model)
        except Exception as e:
            log.warning(
                f'Failed to warm pattern "{pattern}" on {backend.config.address}: {e}'
            )

    def ask(
        self,
        input: str | dict,
        pattern: str,
        model: str = None,
        system_prompt: str = None,
        history_name: str = None,
        include_system_prompt: bool = True,
        include_base_examples: bool = True,
        n_hist: int = 0,
        n_example: int = 0,
        min_d: float = None,
        use_cache: bool = True,
        cache_slot: int = None,
        schema: BaseModel = None,
        stop: List[str] = [],
        custom_history: List[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Ask the LLM a question or generate a response.

        Parameters:
        - input (str | dict): The input to the LLM.
        - pattern (str): The pattern to use for generating the response.
        - model (str): The model to use for generating the response.
        - system_prompt (str): The system prompt to provide to the LLM.
        - history_name (str): The name of the history to use for generating the response.
        - include_system_prompt (bool): Whether to include the system message.
        - include_base_examples (bool): Whether to include the base examples.
        - n_hist (int): The number of historical examples to load from the database.
        - n_example (int): The number of examples to load from the database.
        - min_d (float): The minimum distance between the input and the examples.
        - use_cache (bool): Whether to use the cache for the response.
        - cache_slot (int): Deprecated and ignored. Cache slots are managed
          internally by the backend.
        - schema (BaseModel): The schema to use for the response.
        - stop (List[str]): Strings to stop the response generation.
        - custom_history (List[dict]): Custom conversation history to use instead of database retrieval.
          Format: [{"input": str|dict, "response": dict, "metadata": dict}, ...]
          When provided, n_hist is ignored and database history lookup is bypassed.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - dict: The response from the LLM.
        """

        if cache_slot is not None:
            log.warning(
                "The cache_slot parameter is deprecated and ignored. Cache "
                "slots are managed internally by the backend."
            )

        metadata = self.db.get_metadata(pattern)

        if not input or (not isinstance(input, str) and not isinstance(input, dict)):
            raise ValueError(
                f"Input must be a string or a dictionary, got \"{type(input).__name__}\""
            )

        if isinstance(input, dict):
            input_key = metadata.get("input_key", "input")
            if not path_exists(input, input_key):
                raise ValueError(
                    f"Input dict must contain \"{input_key}\" key/path, got {list(input.keys())}"
                )

        output = self._get_response(
            input=input,
            pattern=pattern,
            model=model,
            system_prompt=system_prompt,
            history_name=history_name,
            include_system_prompt=include_system_prompt,
            include_base_examples=include_base_examples,
            n_hist=n_hist,
            n_example=n_example,
            min_d=min_d,
            use_cache=use_cache,
            schema=schema,
            stop=stop,
            custom_history=custom_history,
            **kwargs,
        )

        return output

    def rerank(
        self,
        input: str,
        model: str,
        documents: List[str] = [],
        n: int = DEFAULT_RESULT_COUNT,
    ) -> List[dict]:
        """
        Rerank a query using the reranking model.

        Parameters:
        - input (str): The input to the LLM.
        - model (str): The model to use for reranking.
        - documents (list[str]): The documents to rerank.
        - n (int, optional): The number of documents to rerank. Defaults to 3.

        Returns:
        - dict: The response from the LLM.
        """

        if model not in self.models:
            raise ValueError(
                f'Model "{model}" not found. Make sure to connect the model first using the `llm.connect_model()` method.'
            )

        if not self.models[model].is_rerank:
            raise ValueError(
                f'Model "{model}" is not a reranking model. Make sure to use the `llm.connect_model()` to connect to a reranking model.'
            )

        if not all(type(document) == str for document in documents):
            raise ValueError(f"All documents must be strings. Got {documents}")

        raw_results = None
        last_error = None
        for backend in self._get_backends(model):
            if not self._check_health(backend):
                log.info(
                    f"Skipping unhealthy backend {backend.config.address} "
                    f"for reranking"
                )
                continue

            try:
                raw_results = backend.rerank(input, documents)
                self._mark_healthy(backend)
                break
            except requests.exceptions.RequestException as e:
                self._mark_unhealthy(backend)
                log.warning(
                    f"Rerank backend {backend.config.address} failed, "
                    f"trying next: {e}"
                )
                last_error = e
                continue
        else:
            raise ValueError(
                f"All rerank backends failed for model '{model}': {last_error}"
            )

        results = []
        for result in raw_results:
            results.append(
                {
                    "index": result["index"],
                    "document": documents[result["index"]],
                    "score": sigmoid(result["relevance_score"]),
                    "raw_score": result["relevance_score"],
                }
            )

        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        return sorted_results[:n]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
