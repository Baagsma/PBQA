import json
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

        self.models = {}

        self.cache_ids = {}

        self.db.create_collection("history")

        self.cf = {
            "llm_response": True,
            "formatting_settings": True,
            "prompt_text": True,
        }

    def connect_to_server(
        self,
        port: int,
        model: str,
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
        - port (int): The port of the LLM server.
        - model (str): The model to use for generating responses.
        - host (str): The host of the LLM server.
        - temperature (float): The temperature to use for generating responses.
        - min_p (float): The minimum probability to use for generating responses.
        - top_p (float): The top probability to use for generating responses.
        - max_tokens (int): The maximum number of tokens to use for generating responses.
        - stop (List[str]): The stop words to use for generating responses.

        Returns:
        - dict[str, str]: The model properties.
        """

        if not host:
            host = self.host
        if not host:
            raise ValueError("Failed to connect to LLM server. No host provided.")

        if self.poke_server(host, port):
            log.info(f"Connected to LLM server at {host}:{port}")
            log.info(f"\tmodel: {model}")
        else:
            raise ValueError(
                f"Failed to connect to LLM server at {host}:{port}. Ensure the server is running."
            )

        self.models[model] = {
            "host": host,
            "port": port,
            "temperature": temperature,
            "min_p": min_p,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop if stop else [],
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
        response: str,
        model: str,
        external: dict[str, str] = {},
        include: List[str] = [],
        exclude: List[str] = [],
        use_cache: bool = True,
        n_hist: int = 0,
        n_example: int = 0,
        grammar: str = None,
        **kwargs,
    ) -> json:
        """
        Get a response from the LLM server.

        Parameters:
        - input (str): The input to the LLM.
        - response (str): The response to generate.
        - external (dict[str, str]): External properties to include in the response.
        - exclude (List[str]): Properties to exclude from the response.
        - use_cache (bool): Whether to cache the exchange.
        - n_hist (int): The number of historical examples to load from the database.
        - n_example (int): The number of examples to load from the database.
        - grammar (str): The grammar to use for the response.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - json: The response from the LLM server.
        """

        prev = time()
        log.info(f"Generating response from LLM")

        metadata = self.db.get_metadata(response)

        if not_present_properties := [
            e for e in exclude if e not in metadata["properties"]
        ]:
            raise ValueError(
                f"Properties {not_present_properties} to exclude not found in response {response}"
            )

        messages = self._format_messages(
            input=input,
            external=external,
            response=response,
            include=include,
            exclude=exclude,
            include_base_examples=use_cache,
            n_hist=n_hist,
            n_example=n_example,
            **kwargs,
        )

        grammar = (
            self._format_master_grammar(
                response=response,
                exclude=exclude,
            )
            if not grammar
            else grammar
        )

        model_defaults = self.models[model]

        parameters = {
            "host": model_defaults["host"],
            "port": model_defaults["port"],
            "stop": model_defaults["stop"],
            "temperature": model_defaults["temperature"],
            "min_p": model_defaults["min_p"],
            "top_p": model_defaults["top_p"],
            **kwargs,  # This will override the defaults if any of these keys are present in kwargs
        }

        data = {
            "model": model,
            "id_slot": self._get_cache_id(
                response,
                model,
            ),
            "cache_prompt": use_cache,
            "messages": messages,
            "grammar": grammar,
            "stop": parameters["stop"],
            "temperature": parameters["temperature"],
            "min_p": parameters["min_p"],
            "top_p": parameters["top_p"],
            "max_tokens": model_defaults["max_tokens"],
        }

        url = f"http://{parameters['host']}:{parameters['port']}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}

        try:
            llm_response = requests.post(url, headers=headers, data=json.dumps(data))
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Request to LLM failed: {str(e)}\n\nEnsure the llama.cpp server is running (python3 server/run.py)."
            )

        llm_response = llm_response.json()

        log.info(f"response:\n{yaml.dump(llm_response, default_flow_style=False)}")

        log.info(
            f"Generated response in {time() - prev:.3f} s ({llm_response['usage']['completion_tokens']/(time() - prev):.1f} t/s)"
        )
        return llm_response

    def _format_messages(
        self,
        response: str,
        include: List[str] = [],
        exclude: List[str] = [],
        input: str = None,
        external: dict[str, str] = {},
        include_base_examples: bool = True,
        system_message: bool = True,
        n_example: int = 0,
        n_hist: int = 0,
        hist_duration: int = DEFAULT_HIST_DURATION,
        max_d: float = None,
        user_name: str = DEFAULT_USER_NAME,
        assistant_name: str = DEFAULT_ASSISTANT_NAME,
        **kwargs,
    ) -> list[dict[str, str]]:
        """
        Format the messages for the LLM.

        Parameters:
        - response (str): The response to generate.
        - exclude (List[str]): Properties to exclude from the response.
        - input (str): The input to the LLM.
        - include (List[str]): Properties to include in the response.
        - external (dict[str, str]): External properties to include in the response.
        - include_base_examples (bool): Whether to include the base messages.
        - system_message (bool): Whether to include the system message.
        - n_example (int): The number of examples to load from the database.
        - n_hist (int): The number of historical examples to load from the database.
        - hist_duration (int): The duration of the historical examples to load.
        - max_d (float): The maximum distance to use in the query.
        - user_name (str): The name of the user.
        - assistant_name (str): The name of the assistant.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - list[dict[str, str]]: The formatted messages.
        """

        metadata = self.db.get_metadata(response)
        properties = metadata["properties"]
        log.info(f"n_example: {n_example}")
        log.info(f"n_hist: {n_hist}")
        log.info(f"metadata:\n{yaml.dump(metadata, default_flow_style=False)}")

        def format(
            responses: list[dict],
            properties: List[str] = properties,
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
                user_properties = ["input"] + [
                    item
                    for item in include
                    if item in external_keys_list and item not in exclude
                ]
                assistant_properties = [
                    item
                    for item in include
                    if item in properties and item not in exclude
                ]
            else:
                user_properties = ["input"] + [
                    item for item in external_keys_list if item not in exclude
                ]
                assistant_properties = [
                    item
                    for item in properties
                    if item not in external_keys_list and item not in exclude
                ]

            def format_role(
                role: str,
                response: dict,
                properties: List[str],
            ) -> dict[str, str]:
                if len(properties) == 1:
                    return {"role": role, "content": response[properties[0]]}
                else:
                    return {
                        "role": role,
                        "content": json.dumps(
                            {prop: response[prop] for prop in properties}
                        ),
                    }

            formatted_responses = []

            for response in responses:
                formatted_responses.append(
                    format_role(
                        user,
                        response,
                        user_properties,
                    )
                )

                formatted_responses.append(
                    format_role(
                        assistant,
                        response,
                        assistant_properties,
                    )
                )

            return formatted_responses

        messages = []

        if include_base_examples:
            if system_message and "system_message" in metadata.keys():
                messages += [
                    (
                        {
                            "role": "system",
                            "content": metadata[
                                "system_message"
                            ],  # TODO: MULT When changing responses to examples, change the system message to go up one level in the hierarchy
                        }
                    )
                ]

            n_base_example = metadata["cache"]["n_example"]

            base_examples = self.db.where(
                collection_name=response,
                n=n_base_example,
                base_example={"$eq": True},
            )

            messages += format(base_examples)

        if input:
            properties = [
                prop for prop in metadata["properties"] if prop not in exclude
            ]

            property_filter = (
                {"$or": [{prop: {"$ne": 0}} for prop in properties]}
                if len(properties) > 1
                else {properties[0]: {"$ne": 0}}
            )  # ensure that at least one property from the desired response is present in the example response

            where_filter = {
                "$and": [
                    property_filter,
                    {"base_example": {"$ne": True}},
                ]
            }

            if kwargs:
                where_filter["$and"].append(kwargs)

            examples = self.db.query(
                response,
                input,
                n=n_example,
                max_d=max_d,
                where=where_filter,
            )
            messages += format(examples)

            hist = self.db.where(
                collection_name="history",
                start=time() - hist_duration,
                end=time(),
                n=n_hist,
                base_example={"$ne": True},
            )  # TODO: If len(hist) == n_hist, remove n oldest responses

            input_response = {
                "input": input,
                **external,
                **{
                    prop: ""
                    for prop in properties
                    if prop not in external and prop not in exclude
                },
            }

            hist.append(input_response)

            messages += format(hist)

            messages = messages[:-1]

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
        response: str,
        exclude: List[str] = [],
    ) -> str:
        """
        Format the master grammar for the LLM.

        Parameters:
        - response (str): The response to generate.
        - exclude (List[str]): Properties to exclude from the response.

        Returns:
        - str: The formatted master grammar.
        """

        metadata = self.db.get_metadata(response)

        grammars = {}
        for prop in metadata["properties"]:
            if prop in exclude or (prop in metadata and "external" in metadata[prop]):
                continue
            if prop not in metadata:
                grammars[prop] = None
                continue
            if "grammar" not in metadata[prop]:
                grammars[prop] = None
            else:
                grammars[prop] = metadata[prop]["grammar"]

        if len(grammars) == 0:
            return None

        if len(grammars) == 1:
            return grammars[list(grammars.keys())[0]]

        grammar = 'root ::= "{"'

        for prop, value in grammars.items():
            grammar += f' "\\"{prop}\\": " {prop + "-gram" if value else "string"} ", "'

        grammar = grammar[:-3]  # Remove the trailing comma and space

        # TODO: Improve string grammar
        grammar += """}"

string ::=
    "\\"" (
    [^\\"]
    )* "\\""

"""

        for prop in grammars:
            if not grammars[prop]:
                continue
            grammar += f"{prop + '-gram'}" + grammars[prop].split("root")[1] + "\n\n"

        return grammar

    def _get_cache_id(self, response: str, model: str) -> str:
        moniker = self.get_cache_name(response, model)

        if moniker not in self.cache_ids:
            self.cache_ids[moniker] = len(self.cache_ids)

        return self.cache_ids[moniker]

    def get_cache_name(self, response: str, model: str) -> str:
        return f"{response}_{model}"

    def ask(
        self,
        input: str,
        response: str,
        model: str,
        external: dict[str, str] = {},
        include: List[str] = [],
        exclude: List[str] = [],
        n_hist: int = 0,
        n_example: int = 0,
        cache: bool = True,
        grammar: str = None,
        **kwargs,
    ) -> Union[str, dict]:
        """
        Ask the LLM a question.

        Parameters:
        - input (str): The input to the LLM.
        - response (str): The response to generate.
        - model (str): The model to use for generating the response.
        - external (dict[str, str]): External properties to include in the response.
        - include (List[str]): Properties to include in the response.
        - exclude (List[str]): Properties to exclude from the response.
        - n_hist (int): The number of historical examples to load from the database.
        - n_example (int): The number of examples to load from the database.
        - cache (bool): Whether to cache the exchange.
        - grammar (str): The grammar to use for the response.
        - kwargs: Additional arguments to pass when querying the database.

        Returns:
        - Union[str, dict]: The response from the LLM.
        """

        metadata = self.db.get_metadata(response)

        external_properties = [
            prop
            for prop in metadata["properties"]
            if prop in metadata and "external" in metadata[prop] and prop not in exclude
        ]
        if not set(external_properties).issubset(set(external.keys())):
            raise ValueError(
                f"External properties {external_properties} not found in external properties {external}. Make sure to all external properties are provided, e.g. external={{'location': 'Amsterdam'}}."
            )

        output = self._get_response(
            input=input,
            response=response,
            model=model,
            include=include,
            external=external,
            exclude=exclude,
            n_hist=n_hist,
            n_example=n_example,
            use_cache=cache,
            grammar=grammar,
            **kwargs,
        )

        answer = output["choices"][0]["message"]["content"]

        if len(set(metadata["properties"]) - set(exclude) - set(external.keys())) == 1:
            return answer
        else:
            try:
                return json.loads(answer)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to decode JSON answer:\n\t{answer}\n\nMake sure the correct stop strings are configured for the model {model}."
                )
