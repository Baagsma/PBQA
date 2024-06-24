import json
from json import dumps, loads
import logging
from time import time
from typing import List, Union

import requests
import yaml

from PBQA.db import DB

log = logging.getLogger()


class LLM:
    DEFAULT_HIST_DURATION = 900
    DEFAULT_USER_NAME = "user"
    DEFAULT_ASSISTANT_NAME = "assistant"

    def __init__(
        self,
        db: DB,
        host: str = None,
        log_level: int = logging.WARN,
    ):
        """
        Initialize the LLM (Language Learning Model.

        This function initializes the LLM with the specified model and database.

        Parameters:
        - db (DB): The database to use for storing and retrieving examples.
        - host (str): The host of the LLM server. Can also be passed when connecting model servers.
        """

        logging.basicConfig(level=log_level)

        self.db = db

        self.host = host

        self.models = {}

        self.cache_ids = {}

    def connect_model(
        self,
        model: str,
        port: int,
        host: str = None,
        temperature: float = 1.2,
        min_p: float = 0.07,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        stop: List[str] = None,
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
        - kwargs: Additional default parameters to pass when querying the LLM server.

        Returns:
        - dict[str, str]: The model components.
        """

        if not host:
            host = self.host
        if not host:
            raise ValueError("Failed to connect to LLM server. No host provided.")

        if self.poke_server(host, port):
            log.info(f"Connected to model {model} at {host}:{port}")
        else:
            raise ValueError(
                f"Failed to connect to LLM server at {host}:{port}. Ensure the server is running and the host and port are correct."
            )

        self.models[model] = {
            "host": host,
            "port": port,
            "temperature": temperature,
            "min_p": min_p,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop if stop else [],
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
                f"Failed to connect to LLM server at {host}:{port}. Ensure the server is running."
            )
            return False

    def _get_response(
        self,
        input: str,
        pattern: str,
        model: str,
        external: dict[str, str] = {},
        history_name: str = None,
        system_prompt: str = None,
        include_system_prompt: bool = True,
        include_base_examples: bool = True,
        include: List[str] = [],
        exclude: List[str] = [],
        n_hist: int = 0,
        n_example: int = 0,
        min_d: float = None,
        use_cache: bool = True,
        grammar: str = None,
        stop: List[str] = [],
        **kwargs,
    ) -> json:
        """
        Get a response from the LLM server.

        Parameters:
        - input (str): The input to the LLM.
        - pattern (str): The pattern to use for generating the response.
        - model (str): The model to use for generating the response.
        - external (dict[str, str]): External data to include in the response.
        - history_name (str): The name of the history to use for generating the response.
        - system_prompt (str): The system prompt to provide to the LLM.
        - include_system_prompt (bool): Whether to include the system message.
        - include_base_examples (bool): Whether to include the base examples.
        - include (List[str]): components to include in the response.
        - exclude (List[str]): components to exclude from the response.
        - n_hist (int): The number of historical examples to load from the database.
        - n_example (int): The number of examples to load from the database.
        - min_d (float): The minimum distance between the input and the examples.
        - use_cache (bool): Whether to use the cache for the response.
        - grammar (str): The grammar to use for the response.
        - stop (List[str]): Strings to stop the response generation.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - json: The response from the LLM.
        """

        prev = time()
        log.info(f"Generating response from LLM")

        metadata = self.db.get_metadata(pattern)

        if not_present_components := [
            e for e in exclude if e not in metadata["components"]
        ]:
            raise ValueError(
                f"components {not_present_components} to exclude not found in pattern {pattern}"
            )

        messages = self._format_messages(
            input=input,
            external=external,
            pattern=pattern,
            include=include,
            exclude=exclude,
            history_name=history_name,
            include_base_examples=include_base_examples,
            system_prompt=system_prompt,
            include_system_prompt=include_system_prompt,
            n_hist=n_hist,
            n_example=n_example,
            min_d=min_d,
            **kwargs,
        )

        grammar = grammar or self._format_master_grammar(
            pattern=pattern,
            exclude=exclude,
        )

        model_defaults = self.models[model]

        parameters = {
            **model_defaults,
            **kwargs,  # This will override the defaults if any of these keys are present in kwargs
        }
        parameters["stop"] = model_defaults["stop"] + stop

        log.info(f"Request:\n{yaml.dump(parameters, default_flow_style=False)}")

        data = {
            "model": model,
            "id_slot": self._get_cache_id(
                pattern,
                model,
            ),
            "cache_prompt": use_cache,
            "messages": messages,
            "grammar": grammar,
            **parameters,
        }

        url = f"http://{parameters['host']}:{parameters['port']}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}

        try:
            llm_response = requests.post(url, headers=headers, data=dumps(data))
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Request to LLM failed: {str(e)}\n\nEnsure the llama.cpp server is running (python3 server/run.py)."
            )

        llm_response = llm_response.json()

        log.info(f"Response:\n{yaml.dump(llm_response, default_flow_style=False)}")

        log.info(
            f"Generated response in {time() - prev:.3f} s ({llm_response['usage']['completion_tokens']/(time() - prev):.1f} t/s)"
        )
        return llm_response

    def _format_messages(
        self,
        pattern: str,
        include: List[str] = [],
        exclude: List[str] = [],
        input: str = None,
        external: dict[str, str] = {},
        history_name: str = None,
        include_base_examples: bool = True,
        system_prompt: str = None,
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
        - exclude (List[str]): components to exclude from the response.
        - input (str): The input to the LLM.
        - include (List[str]): components to include in the response.
        - external (dict[str, str]): External components to include in the response.
        - history_name (str): The name of the history to use for generating the response.
        - include_base_examples (bool): Whether to include the base messages.
        - system_prompt (str): The system prompt to provide to the LLM.
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
        components = metadata["components"]
        log.info(yaml.dump(metadata, default_flow_style=False))

        def format(
            responses: list[dict],
            components: List[str] = components,
            external: dict[str, str] = external,
            include: List[str] = include,
            exclude: List[str] = exclude,
            user: str = user_name,
            assistant: str = assistant_name,
        ) -> list[dict[str, str]]:
            if not responses:
                return []

            external_keys_list = list(external.keys())

            if include:
                components = [item for item in components if item in include]

            user_components = ["input"] + [
                item
                for item in components
                if item in external_keys_list and item not in exclude
            ]
            assistant_components = [
                item
                for item in components
                if item not in external_keys_list and item not in exclude
            ]

            def format_role(
                role: str,
                response: dict,
                components: List[str],
            ) -> dict[str, str]:
                if len(components) == 1:
                    return {"role": role, "content": response[components[0]]}
                else:
                    return {
                        "role": role,
                        "content": dumps({comp: response[comp] for comp in components}),
                    }

            formatted_responses = []

            for response in responses:
                formatted_responses.append(
                    format_role(
                        user,
                        response,
                        user_components,
                    )
                )

                formatted_responses.append(
                    format_role(
                        assistant,
                        response,
                        assistant_components,
                    )
                )

            return formatted_responses

        messages = []

        if include_system_prompt and (
            "system_prompt" in metadata.keys() or system_prompt
        ):
            messages += [
                (
                    {
                        "role": "system",
                        "content": system_prompt or metadata["system_prompt"],
                    }
                )
            ]

        if include_base_examples:
            n_base_example = 50

            base_examples = self.db.where(
                collection_name=pattern,
                n=n_base_example,
                base_example={"eq": True},
            )

            messages += format(base_examples)

        examples = []
        if input:
            components = [
                comp for comp in metadata["components"] if comp not in exclude
            ]

            component_filter = (
                {"_or": [{comp: {"ne": 0}} for comp in components]}
                if len(components) > 1
                else {components[0]: {"ne": 0}}
            )  # Ensure that at least one component from the pattern is present in the example response

            where_filter = {
                "_and": [
                    component_filter,
                    {"base_example": {"ne": True}},
                ]
            }

            if kwargs:
                where_filter["_and"].append(kwargs)

            examples = self.db.query(
                pattern,
                input,
                n=n_example,
                min_d=min_d,
                **where_filter,
            )
            messages += format(examples)

        hist = []
        if n_hist:
            hist = self.db.where(
                collection_name=history_name or pattern,
                start=time() - hist_duration,
                end=time(),
                n=n_hist,
                base_example={"ne": True},
            )  # TODO: If len(hist) == n_hist, remove n oldest responses

            hist.reverse()

        input_response = {
            "input": input,
            **external,
            **{
                comp: ""
                for comp in components
                if comp not in external and comp not in exclude
            },
        }

        hist.append(input_response)

        messages += format(hist)

        messages = messages[:-1]  # The format method adds a vestigial assistant message

        log.info(f"Examples: {len(examples)}/{n_example}")
        log.info(f"History: {len(hist) - 1}/{n_hist}")

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

    def _format_master_grammar(
        self,
        pattern: str,
        exclude: List[str] = [],
    ) -> str:
        """
        Format the master grammar for the LLM.

        Parameters:
        - pattern (str): The pattern to use for generating the response.
        - exclude (List[str]): components to exclude from the pattern.

        Returns:
        - str: The formatted master grammar.
        """

        metadata = self.db.get_metadata(pattern)

        grammars = {}

        for comp in metadata["components"]:
            if (
                comp in exclude
                or not metadata[comp]
                or metadata[comp].get("external", False)
            ):
                continue

            grammars[comp] = metadata[comp].get("grammar", None)

        if len(grammars) == 0:
            return None

        if len(grammars) == 1:
            return grammars[
                list(grammars.keys())[0]
            ]  # If there is only one component, return the grammar for that component

        grammar = 'root ::= "{"'

        for comp, value in grammars.items():
            grammar += f' "\\"{comp}\\": " {comp + "-gram" if value else "string"} ", "'

        grammar = grammar[:-3]  # Remove the trailing comma and space

        # TODO: Improve string grammar
        grammar += """}"

string ::=
    "\\"" (
    [^\\"]
    )* "\\""

"""

        for comp in grammars:
            if not grammars[comp]:
                continue
            grammar += f"{comp + '-gram'}" + grammars[comp].split("root")[1] + "\n\n"

        return grammar

    def _get_cache_id(self, pattern: str, model: str) -> str:
        moniker = self.get_cache_name(pattern, model)

        if moniker not in self.cache_ids:
            self.cache_ids[moniker] = len(self.cache_ids)

        return self.cache_ids[moniker]

    def get_cache_name(self, pattern: str, model: str) -> str:
        return f"{pattern}_{model}"

    def ask(
        self,
        input: str,
        pattern: str,
        model: str,
        external: dict[str, str] = {},
        history_name: str = None,
        system_prompt: str = None,
        include_system_prompt: bool = True,
        include_base_examples: bool = True,
        include: List[str] = [],
        exclude: List[str] = [],
        n_hist: int = 0,
        n_example: int = 0,
        min_d: float = None,
        use_cache: bool = True,
        grammar: str = None,
        stop: List[str] = [],
        **kwargs,
    ) -> dict:
        """
        Ask the LLM a question.

        Parameters:
        - input (str): The input to the LLM.
        - pattern (str): The pattern to use for generating the response.
        - model (str): The model to use for generating the response.
        - external (dict[str, str]): External data to include in the response.
        - history_name (str): The name of the history to use for generating the response.
        - system_prompt (str): The system prompt to provide to the LLM.
        - include_system_prompt (bool): Whether to include the system message.
        - include_base_examples (bool): Whether to include the base examples.
        - include (List[str]): components to include in the response.
        - exclude (List[str]): components to exclude from the response.
        - n_hist (int): The number of historical examples to load from the database.
        - n_example (int): The number of examples to load from the database.
        - min_d (float): The minimum distance between the input and the examples.
        - use_cache (bool): Whether to use the cache for the response.
        - grammar (str): The grammar to use for the response.
        - stop (List[str]): Strings to stop the response generation.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - Union[str, dict]: The response from the LLM.
        """

        metadata = self.db.get_metadata(pattern)

        external_components = [
            comp
            for comp in metadata["components"]
            if metadata[comp]
            and comp not in exclude
            and metadata[comp].get("external", False)
        ]
        if not set(external_components).issubset(set(external.keys())):
            raise ValueError(
                f"External components {external_components} not found in external components {external}. Make sure to all external components are provided, e.g. external={{'location': 'Amsterdam'}}."
            )

        output = self._get_response(
            input=input,
            pattern=pattern,
            model=model,
            include=include,
            external=external,
            history_name=history_name,
            system_prompt=system_prompt,
            include_system_prompt=include_system_prompt,
            include_base_examples=include_base_examples,
            exclude=exclude,
            n_hist=n_hist,
            n_example=n_example,
            min_d=min_d,
            use_cache=use_cache,
            grammar=grammar,
            stop=stop,
            **kwargs,
        )

        answer = output["choices"][0]["message"]["content"]

        if (
            len(
                component := set(metadata["components"])
                - set(exclude)
                - set(external.keys())
            )
            == 1
        ):
            return loads(
                dumps({component.pop(): answer})
            )  # When there is only one component, the model output is a string without a grammar for quality and speed, so the output has to be parsed into a dictionary manually
        else:
            try:
                return loads(answer)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to decode JSON answer:\n\t{answer}\n\nMake sure the correct stop strings are configured for the model {model}."
                )
