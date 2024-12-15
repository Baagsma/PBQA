import json
import logging
import math
from time import time
from typing import List

import requests
import yaml
from pydantic import BaseModel

from PBQA.db import DB

log = logging.getLogger()


class LLM:
    DEFAULT_HIST_DURATION = 900
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

        self.cache_slots = {}
        self.models = {}
        self.pattern_models = {}

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
        **kwargs,
    ) -> dict[str, str]:
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
        - kwargs: Additional default parameters to pass when querying the LLM server.

        Returns:
        - dict[str, str]: The model components.
        """

        if not host:
            host = self.host
        if not host:
            raise ValueError("Failed to connect to LLM server. No host provided.")

        props = self.get_props(host, port)
        if props == {}:
            raise ValueError(f"Failed to connect to LLM server at {host}:{port}")

        is_rerank = self.is_rerank(host, port)

        store_cache = store_cache and not is_rerank and self.can_store_cache(host, port)

        log.info(f'Connected to model "{model}" at {host}:{port}')

        self.models[model] = {
            "host": host,
            "port": port,
            "is_rerank": is_rerank,
            "temperature": temperature,
            "min_p": min_p,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
            "total_slots": props.get("total_slots", 1096),
            "store_cache": store_cache,
            **kwargs,
        }

        return self.models[model]

    @staticmethod
    def poke_server(host: str, port: int) -> bool:
        url = f"http://{host}:{port}"

        try:
            requests.get(url + "/health")
            return True
        except requests.exceptions.RequestException as e:
            log.warn(
                f"Failed to connect to LLM server at {host}:{port}. Ensure the server is running and the host and port are correct."
            )
            return False

    @staticmethod
    def can_store_cache(host: str, port: int) -> bool:
        url = f"http://{host}:{port}"

        try:
            response = requests.post(
                url + "/slots/0?action=restore",
                json={"filename": "zppivzcjfxvavwyqxse.bin"},
            ).json()
            if "error" in response:
                if response["error"]["code"] == 400:
                    log.info(
                        f"Connection to slot saving endpoint at {host}:{port} successful"
                    )
                    return True
                log.warn(
                    f"Failed to connect to slot saving endpoint at {host}:{port} with error {response['error']['code']}: {response['error']['message']}"
                )
            return False
        except requests.exceptions.RequestException as e:
            log.info(
                f"Failed to connect to slot saving endpoint at {host}:{port}. Disabling cache saving."
            )
            return False

    @staticmethod
    def is_rerank(host: str, port: int) -> bool:
        url = f"http://{host}:{port}"

        try:
            response = requests.post(
                url + "/v1/rerank",
                json={"query": "test", "documents": ["test"]},
            ).json()
            log.warn(f"Response: {response}")
            if "error" in response:
                log.warn(
                    f"Failed to connect to reranking endpoint at {host}:{port} with error {response['error']['code']}: {response['error']['message']}"
                )
                return False
            log.info(f"Model at {host}:{port} supports reranking")
            return True
        except requests.exceptions.RequestException as e:
            log.warn(f"Model at {host}:{port} does not support reranking")

    @staticmethod
    def get_props(host: str, port: int) -> dict:
        url = f"http://{host}:{port}"

        try:
            response = requests.get(url + "/props")
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Failed to get properties from LLM server at {host}:{port}. Ensure the server is running and the host and port are correct."
            )

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
        cache_slot: int = None,
        schema: BaseModel = None,
        stop: List[str] = [],
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
        - cache_slot (int): The cache slot to use for the response.
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

        if self.models[model].get("is_rerank", False):
            raise ValueError(
                f'Model "{model}" is a reranking model. Make sure to use the `llm.rerank()` method instead of `llm.ask()`.'
            )

        prev = time()
        log.info(f"Generating response from LLM")

        if cache_slot is None or cache_slot >= self.models[model].get(
            "total_slots", 1096
        ):
            if cache_slot:
                log.warn(
                    f"Provided cache slot {cache_slot} exceeds the maximum number of cache slots {self.models[model].get('total_slots', 1096)} or pattern \"{pattern}\" and model \"{model}\". Using the last slot instead."
                )
            cache_slot = self._get_cache_slot(pattern, model)

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
            **kwargs,
        )

        metadata = self.db.get_metadata(pattern)

        # If the schema consists of a single str component, pass None instead of the schema
        schema = schema or metadata["schema"]
        if (
            len(schema["properties"]) == 1
            and (prop_name := list(schema["properties"].keys())[0])
            and schema["properties"][prop_name]["type"] == "string"
        ):
            log.info(
                f"Schema consists of a single string component ({prop_name}). Passing None instead of the schema."
            )
            schema = None

        parameters = {**self.models[model], **kwargs}

        data = {
            "model": model,
            "id_slot": cache_slot,
            "cache_prompt": use_cache,
            "messages": messages,
            **({"json_schema": schema} if schema else {}),
            "stop": parameters.get("stop", []) + stop,
            **parameters,
        }

        if parameters["store_cache"]:
            self.load_cache(model, pattern, cache_slot)

        log.info(
            f"Performing query ({pattern}-{model}) at {parameters['host']}:{parameters['port']} ID slot {cache_slot}"
        )

        try:
            then = time()
            url = (
                f"http://{parameters['host']}:{parameters['port']}/v1/chat/completions"
            )
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer no-key",
            }
            raw_response = requests.post(
                url, headers=headers, data=json.dumps(data)
            ).json()
            if "error" in raw_response:
                raise ValueError(f"LLM error:\n{json.dumps(raw_response, indent=4)}")
            content = raw_response["choices"][0]["message"]["content"]
            llm_response = json.loads(content) if schema else content
            log.info(f"Response:\n{json.dumps(llm_response, indent=4)}")
            response_time = time() - then
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Request to LLM failed: {str(e)}\n\nEnsure the llama.cpp server is running."
            )

        if parameters["store_cache"]:
            self.save_cache(model, pattern, cache_slot)

        return {
            "input": input,
            "response": llm_response if schema else {prop_name: llm_response},
            "metadata": {
                "total_time": response_time,
                **raw_response["usage"],
            },
        }

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

            messages = []
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

        if include_base_examples:
            base_examples = self.db.where(
                collection_name=pattern,
                n=999,
                base_example=True,
            )
            base_examples.reverse()
            messages += format(base_examples)

        log.info(f"Base examples: {len(base_examples)}")

        if input:
            query_input = input
            if type(input) == dict:
                if metadata.get("input_key", "input") not in input:
                    raise ValueError(
                        f"Input dict must contain {metadata['input_key']} key, got {input.keys()}"
                    )
                query_input = input[metadata["input_key"]]

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
        if n_hist:
            hist = self.db.where(
                collection_name=history_name or pattern,
                start=time() - hist_duration,
                end=time(),
                n=n_hist,
                base_example={"ne": True},
            )  # TODO: If len(hist) == n_hist, remove n oldest responses - something something modulo

            # Assert that the history is sorted by time_added
            assert hist == sorted(
                hist, key=lambda x: x["metadata"]["time_added"], reverse=True
            ), f"Expected the history to be sorted by time_added, got {json.dumps(hist, indent=4)}"

            hist.reverse()

            messages += format(hist)

        log.info(f"History: {len(hist)}/{n_hist}")

        messages.append(format_role(user_name, input))

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
    ) -> int:
        """
        Link a pattern-model pair to a cache slot.

        Parameters:
        - pattern (str): The pattern to assign the cache slot to.
        - model (str): The model to assign the cache slot to.
        - cache_slot (int): The cache slot to assign to the pattern and model

        Returns:
        - int: The cache slot.
        """

        if pattern not in self.db.get_patterns():
            raise ValueError(
                f'Pattern "{pattern}" not found. Make sure to load the pattern first using the `db.load_pattern()` method.'
            )
        if model not in self.models:
            raise ValueError(
                f'Model "{model}" not found. Make sure to connect the model first using the `llm.connect_model()` method.'
            )

        total_slots = self.models[model].get("total_slots", 1096)

        if cache_slot is None or cache_slot >= total_slots:
            if cache_slot is not None:
                log.warn(
                    f"Provided cache slot {cache_slot} exceeds the total number of cache slots {total_slots} for pattern-model pair {pattern}-{model}. Using an auto-assigned slot instead."
                )
            result_slot = self._get_cache_slot(pattern, model)
        else:
            result_slot = cache_slot

        if model not in self.cache_slots:
            self.cache_slots[model] = {}
        self.cache_slots[model][pattern] = result_slot

        log.info(
            f"Assigned pattern-model pair {pattern}-{model} to cache slot {result_slot}"
        )

        self.pattern_models[pattern] = model

        return result_slot

    def _get_cache_slot(self, pattern: str, model: str) -> int:
        total_slots = self.models[model].get("total_slots", 1096)

        if model not in self.cache_slots:
            self.cache_slots[model] = {}

        if pattern not in self.cache_slots[model]:
            # Find the lowest available slot for this model
            for slot in range(total_slots):
                if slot not in self.cache_slots[model].values():
                    self.cache_slots[model][pattern] = slot
                    break
            else:
                # If no slots are available for this model, use the last slot
                self.cache_slots[model][pattern] = total_slots - 1

        return self.cache_slots[model][pattern]

    def load_cache(self, model: str, pattern: str, slot: int):
        url = f"http://{self.models[model]['host']}:{self.models[model]['port']}"

        try:
            response = requests.post(
                url + f"/slots/{slot}?action=restore",
                json={"filename": f"{pattern}-{model}.bin"},
            ).json()
            if "error" in response:
                if response["error"]["code"] == 400:
                    log.info(f"Cache for {pattern}-{model} not found")
                    return
                log.warn(
                    f"Failed to load cache for {pattern}-{model} to slot {slot} with error {response['error']['code']}: {response['error']['message']}"
                )
                return
            log.info(f"Loaded cache for {pattern}-{model} to slot {slot}")
        except:
            log.warn(f"Failed to load cache for {pattern}-{model} to slot {slot}")

    def save_cache(self, model: str, pattern: str, slot: int):
        url = f"http://{self.models[model]['host']}:{self.models[model]['port']}"

        try:
            requests.post(
                url + f"/slots/{slot}?action=save",
                json={"filename": f"{pattern}-{model}.bin"},
            )
            log.info(f"Saved cache for {pattern}-{model} to slot {slot}")
        except:
            log.warn(f"Failed to save cache for {pattern}-{model} to slot {slot}")

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
        - cache_slot (int): The cache slot to use for the response.
        - schema (BaseModel): The schema to use for the response.
        - stop (List[str]): Strings to stop the response generation.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - dict: The response from the LLM.
        """

        metadata = self.db.get_metadata(pattern)

        if (
            not input
            or not isinstance(input, str)
            and not isinstance(input, dict)
            or (
                isinstance(input, dict) and metadata.get("input_key", None) not in input
            )
        ):
            raise ValueError(
                f'Input must be a string or a dictionary with a key named "{metadata.get('input_key', 'input')}", got "{input}"'
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
            cache_slot=cache_slot,
            schema=schema,
            stop=stop,
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

        if not self.models[model].get("is_rerank", False):
            raise ValueError(
                f'Model "{model}" is not a reranking model. Make sure to use the `llm.connect_model()` to connect to a reranking model.'
            )

        if not all(type(document) == str for document in documents):
            raise ValueError(f"All documents must be strings. Got {documents}")

        try:
            url = f"http://{self.models[model]['host']}:{self.models[model]['port']}"
            response = requests.post(
                url + "/v1/rerank",
                json={"query": input, "documents": documents},
            ).json()
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Request to reranking server at {self.models[model]['host']}:{self.models[model]['port']} failed: {str(e)}"
            )

        results = []
        for result in response["results"]:
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
