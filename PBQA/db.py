import json
import logging
import os
import uuid
from hashlib import sha256
from time import time
from typing import List
from pydantic import BaseModel

import requests
import yaml
from dateutil.parser import parse
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

log = logging.getLogger()


class DB:
    """A wrapper for the Qdrant client with some added functionality for ease of use."""

    DEFAULT_ENCODER = "all-MiniLM-L6-v2"
    DEFAULT_RESULT_COUNT = 5
    DEFAULT_MAX_RESULT_COUNT = 100
    DEFAULT_METADATA_COLLECTION_NAME = "metadata"

    def __init__(
        self,
        path: str = None,
        host: str = None,
        port: int = None,
        encoder: str = DEFAULT_ENCODER,
        metadata_collection_name: str = DEFAULT_METADATA_COLLECTION_NAME,
        reset: bool = False,
        auto_update: bool = False,
    ):
        """
        Initialize the DB client.

        The database can be initialized with either a path to a Qdrant database or a host and port for a remote Qdrant server.

        Parameters:
        - path (str): The path to the database.
        - host (str): The host of the Qdrant server.
        - port (int): The port of the Qdrant server.
        - encoder (str): The name of the SentenceTransformer model to use for encoding documents.
        - metadata_collection_name (str, optional): The name of the collection to store metadata in. Defaults to "metadata".
        - reset (bool, optional): Whether to reset the database. Defaults to False. If True, the collections specified in the metadata collection will be deleted.
        - auto_update (bool, optional): Whether to automatically update the database when the patterns change. Defaults to False. Warning: Any changes to the examples, model, or system prompt will completely overwrite the collection.
        """

        self.use_remote = False
        if host and port:
            try:
                requests.get(f"http://{host}:{port}")
            except requests.exceptions.RequestException as e:
                raise ValueError(
                    f"Failed to connect to Qdrant server at {host}:{port}. Ensure the server is running and the host and port are correct."
                )
            self.client = QdrantClient(host=host, port=port)
            self.use_remote = True
            log.info(f"Connected to Qdrant server at {host}:{port}")
        elif path:
            self.client = QdrantClient(path=path)
            log.info(f"Connected to Qdrant database at {path}")
        else:
            raise ValueError("No path or host provided for the database.")

        self.host = host
        self.port = port
        self.encoder = SentenceTransformer(encoder)
        self.metadata_collection_name = metadata_collection_name
        self.auto_update = auto_update

        if reset:
            log.info("Resetting db")
            self.reset()

        if metadata_collection_name not in [
            collection.name
            for collection in self.client.get_collections().__dict__["collections"]
        ]:  # Since Qdrant doesn't support metadata for collections, we use a separate collection for metadata
            self.metadata = self.client.create_collection(
                collection_name=metadata_collection_name,
                vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE
                ),
            )

    def load_pattern(
        self,
        schema: BaseModel,
        examples: str | list[dict] = None,
        system_prompt: str = None,
        collection_name: str = None,
        input_key: str = "input",
        **kwargs,
    ):
        pattern_name = schema.__name__.lower()
        collection_name = collection_name or pattern_name

        log.info(f"Creating collection {collection_name} for pattern {pattern_name}")

        # Early conversion of file to examples if needed
        if isinstance(examples, str):
            examples = self.load_from_file(examples)

        # Validate examples format
        if examples and not (
            isinstance(examples, list)
            and all(
                isinstance(item, dict) and "user" in item and "assistant" in item
                for item in examples
            )
        ):
            raise ValueError(
                f"Invalid examples format. Expected list of dicts with 'user' and 'assistant' keys, got:\n{json.dumps(examples, indent=4)}"
            )

        # Generate hash for examples, schema, and system prompt
        hash = sha256(
            json.dumps(
                (examples, schema.model_json_schema(), system_prompt, input_key)
            ).encode()
        ).hexdigest()

        # Handle existing collection with different hash
        if collection_name in self.get_collections():
            stored_pattern_name = self.get_metadata(collection_name=collection_name)[
                "pattern_name"
            ]

            # Verify pattern name matches
            if pattern_name != stored_pattern_name:
                raise ValueError(
                    f'Collection "{collection_name}" exists with different pattern name "{stored_pattern_name}". Delete collection first to overwrite'
                )

            stored_hash = self.get_metadata(collection_name=collection_name)["hash"]
            if hash == stored_hash:
                log.info(
                    f'Pattern "{pattern_name}" already loaded into collection "{collection_name}".'
                )
                return

            # Handle hash mismatch
            if not self.auto_update:
                log.warn(
                    f'Pattern "{pattern_name}" was updated. To update, set auto_update=True or manually delete and reload collection.'
                )
                return

            log.info(
                f'Pattern "{pattern_name}" was updated. Updating collection "{collection_name}"'
            )
            self.delete_collection(collection_name)

        # Create new collection
        self.create_collection(
            collection_name=collection_name,
            metadata={
                "pattern_name": pattern_name,
                "schema": schema.model_json_schema(),
                "system_prompt": system_prompt,
                "input_key": input_key,
                "hash": hash,
                "time_added": None,
                **kwargs,
            },
        )

        if examples:
            log.info(f'Loading examples into collection "{collection_name}"')
            prev = time()

            for example in examples:
                # Validate example against model schema
                try:
                    schema.model_validate(example["assistant"])
                except Exception as e:
                    raise ValueError(
                        f"Example doesn't match model schema:\n{json.dumps(example['assistant'], indent=4)}\n\nSchema:\n{json.dumps(schema.model_json_schema(), indent=4)}"
                    ) from e

                self.add(
                    input=example["user"],
                    collection_name=collection_name,
                    base_example=True,
                    **example["assistant"],
                )

            log.info(
                f'Populated collection "{collection_name}" in {time() - prev:.3f} s'
            )

        self.index(collection_name, "metadata.time_added", "float")

    @staticmethod
    def load_from_file(path: str) -> dict:
        """
        Load data from a file.

        This method reads a file and loads its contents into a dictionary. The file can be in either JSON or YAML format.

        Parameters:
        - path (str): The path to the file to read.

        Returns:
        - dict: The contents of the file as a dictionary.
        """

        if path.endswith(".json"):
            with open(path) as f:
                return json.load(f)
        elif path.endswith(".yaml"):
            with open(path) as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise TypeError(
                f'Filetype "{file_extension(path)}" not supported for "{path}"'
            )

    def create_collection(
        self,
        collection_name: str,
        distance: str = "cosine",
        metadata: dict = {},
        **kwargs,
    ):
        """
        Get a collection by name.

        This method retrieves a collection by name. If the collection does not exist, it can be created by setting the `create` parameter to True.

        Parameters:
        - collection_name (str): The name of the collection to retrieve.
        - distance (str, optional): The distance metric to use for the collection. Defaults to "cosine". Must be one of "cosine", "euclid", "dot", or "manhattan".
        - metadata (dict, optional): Metadata to pass when creating the collection.
        - **kwargs: Additional keyword arguments to pass when creating the collection.
        """
        if collection_name in self.get_collections():
            log.info(f'Collection "{collection_name}" already exists')
            return

        log.info(
            f'Creating collection "{collection_name}" with metadata:\n{json.dumps(metadata, indent=4)}'
        )

        config = models.VectorParams(
            size=self.encoder.get_sentence_embedding_dimension(),
            distance=models.Distance[distance.upper()],
        )

        metadata["collection_name"] = collection_name
        self.client.upsert(
            collection_name=self.metadata_collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[0],
                    payload=metadata,
                )
            ],
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=config,
            **kwargs,
        )

    def delete_collection(self, collection_name: str):
        if not self.use_remote:
            raise ValueError(
                "Cannot delete collection locally. Use a remote DB instead."
            )

        if collection_name not in self.get_collections():
            raise ValueError(
                f'Collection "{collection_name}" not found in collections {self.get_collections()}'
            )

        self.client.delete_collection(collection_name=collection_name)

        # Delete the pattern from the metadata
        self.client.delete(
            collection_name=self.metadata_collection_name,
            points_selector=self.filter_to_qdrant_filter(
                {"collection_name": collection_name}
            ),
        )

    def add(
        self,
        input: str | dict,
        collection_name: str,
        time_added: float = None,
        **kwargs,
    ) -> dict:
        """
        Add a document to a collection.

        This method adds a document to a collection. The document should contain an `input` key with the document's input. Additional keys in the dictionary are treated as metadata for the document.

        Parameters:
        - input (str | dict): The document's input. If a dict, it must contain at least an "input" (or whatever the pseudonym is) key, to be used as the embedding vector.
        - collection_name (str): The name of the collection to add the document to.
        - time_added (float, optional): The time the document was added in Unix time. Defaults to time().
        - **kwargs: Additional keyword arguments to pass as metadata for the document.

        Returns:
        - dict: A dictionary representing the added document. The dictionary contains the document's input, id, and metadata.
        """
        if collection_name not in self.get_collections():
            raise ValueError(
                f'Collection "{collection_name}" not found in collections {self.get_collections()}. Make sure to load the pattern first or create the collection manually.'
            )

        metadata = self.get_metadata(collection_name=collection_name)
        input_key = metadata.get("input_key", "input")
        doc_id = str(uuid.uuid4())

        # Extract embedding text
        embedding_text = input[input_key] if isinstance(input, dict) else input
        if isinstance(input, dict) and input_key not in input:
            raise ValueError(f'Input dict must contain "{input_key}" key')

        # Build base document
        doc = {
            "input": input,
            "time_added": time_added or time(),
            **kwargs,
        }

        # Handle schema if present
        if "schema" in metadata:
            schema_keys = self.get_schema_keys(collection_name)
            response_data = {k: v for k, v in kwargs.items() if k in schema_keys}
            doc_metadata = {k: v for k, v in kwargs.items() if k not in schema_keys}

            # Validate response against schema
            if set(response_data.keys()) != set(schema_keys):
                raise ValueError(
                    f"Response {response_data.keys()} doesn't match schema {schema_keys}"
                )

            doc = {
                "input": input,
                "response": response_data,
                "metadata": {
                    **doc_metadata,
                    "id": doc_id,
                    "embedding_text": embedding_text,
                    "time_added": doc["time_added"],
                },
            }

        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector=self.encoder.encode(embedding_text),
                    payload=doc,
                )
            ],
        )

        return doc

    def get_metadata(
        self,
        pattern: str = None,
        collection_name: str = None,
    ) -> dict:
        """
        Get the metadata for a collection.

        This method retrieves the metadata for a collection or pattern.

        Parameters:
        - pattern (str): The name of the pattern to query.
        - collection_name (str): The name of the collection to query. Defaults to None.

        Returns:
        - dict: The metadata for the given collection.
        """

        patterns = self.get_patterns()
        collections = self.get_collections()

        if not pattern and not collection_name:
            raise ValueError("Either pattern or collection_name must be provided.")
        if pattern not in patterns:
            ValueError(
                f'Pattern "{pattern}" not found in patterns {patterns}. Make sure to load the pattern first or create the collection manually.'
            )
        if collection_name not in collections:
            ValueError(
                f'Collection "{collection_name}" not found in collections {collections}. Make sure to load the pattern first or create the collection manually.'
            )

        if pattern:
            metadata = self.client.scroll(
                collection_name=self.metadata_collection_name,
                scroll_filter=self.filter_to_qdrant_filter({"pattern_name": pattern}),
            )[0]
        else:
            metadata = self.client.scroll(
                collection_name=self.metadata_collection_name,
                scroll_filter=self.filter_to_qdrant_filter(
                    {"collection_name": collection_name}
                ),
            )[0]

        if len(metadata) > 1:
            raise ValueError(
                f"Multiple metadata entries found for {'pattern' if pattern else 'collection'} \"{pattern if pattern else collection_name}\". Make sure there is only one {'pattern' if pattern else 'collection'} named \"{pattern if pattern else collection_name}\" in the database."
            )

        return metadata[0].payload

    def get_collections(self) -> List[str]:
        """Get the names of all collections in the database."""
        collections = self.client.scroll(
            collection_name=self.metadata_collection_name,
            limit=99999,
        )[0]
        if len(collections) == 0:
            return []
        return [collection.payload["collection_name"] for collection in collections]

    def get_collection(self, pattern: str) -> str:
        """Get the name of the collection for a given pattern."""
        return self.get_metadata(pattern=pattern)["collection_name"]

    def get_patterns(self) -> List[str]:
        """Get the names of all patterns in the database."""
        collections = self.client.scroll(
            collection_name=self.metadata_collection_name,
            limit=99999,
        )[0]
        collections = [
            collection
            for collection in collections
            if "pattern_name" in collection.payload
        ]
        return (
            []
            if len(collections) == 0
            else [collection.payload["pattern_name"] for collection in collections]
        )

    def n_collections(self):
        """Get the number of collections in the database."""
        try:
            return len(self.get_collections())
        except Exception as e:
            return 0

    def n(self, collection_name: str) -> int:
        """Get the number of documents in a collection."""
        if collection_name not in self.get_collections():
            raise ValueError(f'collection "{collection_name}" not found')

        try:
            return self.client.count(collection_name).count
        except Exception as e:
            return 0

    def reset(self):
        """Reset the database by deleting all collections."""
        if (
            "metadata"
            not in [  # If there is no metadata collection yet, we don't need to delete anything
                collection.name
                for collection in self.client.get_collections().__dict__["collections"]
            ]
        ):
            return
        elif not self.use_remote:
            raise ValueError(
                f"Cannot reset database locally. Use a remote DB instead or reset manually at {self.path}"
            )

        log.warn(f"Resetting database (collections: {self.get_collections()})")

        for collection in self.get_collections():
            self.client.delete_collection(collection_name=collection)
        self.client.delete_collection(collection_name=self.metadata_collection_name)

    def query(
        self,
        collection_name: str,
        input: str,
        n: int = DEFAULT_RESULT_COUNT,
        min_d: int = None,
        **kwargs,
    ) -> List[dict]:
        """
        Query a collection for documents that match the given input.

        This method supports filtering queries by both metadata and document contents. Metadata filters are passed directly as keyword arguments, while the `where_document` parameter is used for document content filters.

        To filter on metadata, supply the metadata field and its desired value directly as keyword arguments. For more complex queries, you can use a dictionary with the following structure:

        {
            <Operator>: <Value>
        }

        Metadata filtering supports the following operators:

        - `eq` - equal to (string, int, float)
        - `ne` - not equal to (string, int, float)
        - `gt` - greater than (int, float)
        - `gte` - greater than or equal to (int, float)
        - `lt` - less than (int, float)
        - `lte` - less than or equal to (int, float)

        Note that metadata filters only search embeddings where the key exists. If a key is not present in the metadata, it will not be returned.

        To filter on document contents, supply a `where_document` filter dictionary to the query. We support two filtering keys: `contains` and `not_contains`.

        You can also use the logical operators `_and` and `_or` to combine multiple filters, and the inclusion operators `in` and `nin` to filter based on whether a value is in or not in a predefined list.

        Parameters:
        - collection_name (str): The name of the pattern or collection to query.
        - input (str): The input to match.
        - n (int, optional): The number of results to return. Defaults to DEFAULT_RESULT_COUNT.
        - min_d (int, optional): The maximum distance for a document to be considered a match. If None, no maximum distance is used.
        - where (dict, optional): A dictionary of metadata filters to apply. Defaults to None.

        Returns:
        - List[dict]: A list of dictionaries, each representing a matching document. Each dictionary contains the document's input, id, distance from the query input, and metadata.
        """

        if n == -1:
            n = self.DEFAULT_MAX_RESULT_COUNT
        elif n < 1:
            return []

        collection_name = self._get_collection_name(collection_name)

        query_filter = self.filter_to_qdrant_filter(kwargs)

        docs = self.client.search(
            collection_name=collection_name,
            query_filter=query_filter,
            limit=n,
            query_vector=self.encoder.encode(input),
            score_threshold=min_d,
        )

        has_schema = "schema" in self.get_metadata(collection_name=collection_name)
        return self._format_docs(docs, has_schema)

    def where(
        self,
        collection_name: str,
        n: int = DEFAULT_RESULT_COUNT,
        start: float = None,
        end: float = None,
        order_by: str = None,
        order_direction: str = "desc",
        **kwargs,
    ) -> List[dict]:
        """
        Query a collection for documents that match the given metadata filters.

        This method supports filtering queries by metadata. Metadata filters are passed directly as keyword arguments.

        To filter on metadata, supply the metadata field and its desired value directly as keyword arguments. For more complex queries, you can use a dictionary with the following structure:

        {
            <Operator>: <Value>
        }

        Metadata filtering supports the following operators:

        - `eq` - equal to (string, int, float)
        - `ne` - not equal to (string, int, float)
        - `gt` - greater than (int, float)
        - `gte` - greater than or equal to (int, float)
        - `lt` - less than (int, float)
        - `lte` - less than or equal to (int, float)

        Note that metadata filters only search embeddings where the key exists. If a key is not present in the metadata, it will not be returned.

        Parameters:
        - collection_name (str): The name of the pattern or collection to query.
        - n (int, optional): The number of results to return. Defaults to DEFAULT_RESULT_COUNT.
        - start (float, optional): The start time for the query. Defaults to None.
        - end (float, optional): The end time for the query. Defaults to None.
        - order_by (str, optional): The key to order the results by. Defaults to "time_added".
        - order_direction (str, optional): The direction to order the results in. Must be "asc" or "desc". Defaults to "desc".
        - **kwargs: Additional keyword arguments to pass as metadata filters.

        Returns:
        - List[dict]: A list of dictionaries, each representing a matching document. Each dictionary contains the document's input, id, and metadata.
        """

        if n == -1:
            n = self.DEFAULT_MAX_RESULT_COUNT
        elif n < 1:
            return []

        collection_name = self._get_collection_name(collection_name)
        metadata = self.get_metadata(collection_name=collection_name)
        has_schema = "schema" in metadata

        time_added_key = "metadata.time_added" if has_schema else "time_added"
        order_by = order_by or time_added_key
        order_obj = models.OrderBy(key=order_by, direction=order_direction)

        if "time_added" in metadata:
            if start and end:
                if "_and" in kwargs:
                    kwargs["_and"].append({time_added_key: {"gte": start}})
                    kwargs["_and"].append({time_added_key: {"lte": end}})
                else:
                    kwargs["_and"] = [
                        {time_added_key: {"gte": start}},
                        {time_added_key: {"lte": end}},
                    ]
            elif start:
                kwargs[time_added_key] = {"gte": start}
            elif end:
                kwargs[time_added_key] = {"lte": end}
        elif order_by in ["time_added", "metadata.time_added"]:
            order_obj = None

        scroll_filter = self.filter_to_qdrant_filter(kwargs)

        docs = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=n,
            order_by=order_obj,
        )[0]

        return self._format_docs(docs, has_schema)

    def _get_collection_name(self, collection_name: str) -> str:
        if collection_name not in self.get_patterns():
            if collection_name not in self.get_collections():
                raise ValueError(
                    f'Neither pattern nor collection named "{collection_name}" found. Make sure to load the pattern first or create the collection manually.'
                )
            else:
                log.info(
                    f'No collection associated with pattern "{collection_name}". Using collection "{collection_name}" instead.'
                )
                collection_name = collection_name
        else:
            collection_name = self.get_collection(collection_name)
        return collection_name

    def _format_docs(self, docs: List[dict], has_schema: bool) -> List[dict]:
        if not docs:
            return []

        if has_schema:
            return [
                {
                    "input": doc.payload["input"],
                    "response": doc.payload["response"],
                    "metadata": {
                        **({"distance": doc.score} if hasattr(doc, "score") else {}),
                        **doc.payload["metadata"],
                    },
                }
                for doc in docs
            ]
        else:
            return [
                {
                    "input": doc.payload["input"],
                    **({"distance": doc.score} if hasattr(doc, "score") else {}),
                    **doc.payload,
                }
                for doc in docs
            ]

    def get_schema_keys(self, collection_name: str) -> List[str]:
        """
        Get the keys of the schema components for a collection.

        This method retrieves the keys of the schema components for a collection.

        Parameters:
        - collection_name (str): The name of the collection to retrieve the keys for.

        Returns:
        - List[str]: The keys of the schema components for the collection.
        """
        schema = self.get_metadata(collection_name=collection_name)["schema"]
        return schema["properties"].keys()

    def index(self, collection_name: str, component: str, type: str):
        """
        Index a pattern component for filtering and ordering.

        This method indexes a pattern component for filtering and ordering (https://qdrant.tech/documentation/concepts/indexing/#payload-index).

        Parameters:
        - collection_name (str): The name of the pattern or collection to index.
        - component (str): The name of the component to index.
        - type (str): The type of index to create. Can be "keyword", "integer", "float", "bool", "geo", "datetime", or "text".
        """
        if not self.use_remote:
            log.warn(f"Indexing is only supported for remote databases.")
            return

        if collection_name not in self.get_collections():
            raise ValueError(f'Pattern "{collection_name}" not found')
        if type not in [
            "keyword",
            "integer",
            "float",
            "bool",
            "geo",
            "datetime",
            "text",
        ]:
            raise ValueError(
                f"Invalid type {type}. Must be one of keyword, integer, float, bool, geo, datetime, or text."
            )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=component,
            field_type=type,
        )
        log.info(f"Indexed {component} in {collection_name} as {type}")

    @staticmethod
    def filter_to_qdrant_filter(filter: dict) -> models.Filter:
        if not filter or not filter.keys():
            return None

        def process_filter(key, value):
            if isinstance(value, dict):
                operator, operand = list(value.items())[0]
            else:
                operator, operand = "eq", value

            if is_valid_date(operand):
                range = models.DatetimeRange()
            else:
                range = models.Range()

            if key == "_and":
                if not isinstance(operand, List):
                    raise ValueError(
                        f"Invalid operand {operand} for operator _and. Should be a list."
                    )
                return models.Filter(
                    must=[process_filter(*list(item.items())[0]) for item in operand]
                )
            elif key == "_or":
                if not isinstance(operand, List):
                    raise ValueError(
                        f"Invalid operand {operand} for operator _or. Should be a list."
                    )
                return models.Filter(
                    should=[process_filter(*list(item.items())[0]) for item in operand]
                )
            elif operator == "eq":
                try:
                    return models.FieldCondition(
                        key=key, match=models.MatchValue(value=operand)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid operand {operand} for operator eq. Filter:\n{json.dumps(filter, indent=4)}\n\nError:\n{str(e)}"
                    )
            elif operator == "ne":
                try:
                    return models.Filter(
                        must_not=[
                            models.FieldCondition(
                                key=key, match=models.MatchValue(value=operand)
                            )
                        ]
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid operand {operand} for operator ne. Filter:\n{json.dumps(filter, indent=4)}\n\nError:\n{str(e)}"
                    )

            elif operator == "gt":
                range.gt = operand
                return models.FieldCondition(key=key, range=range)
            elif operator == "gte":
                range.gte = operand
                return models.FieldCondition(key=key, range=range)
            elif operator == "lt":
                range.lt = operand
                return models.FieldCondition(key=key, range=range)
            elif operator == "lte":
                range.lte = operand
                return models.FieldCondition(key=key, range=range)
            elif operator == "in":
                if not isinstance(operand, list):
                    raise ValueError(
                        f"Invalid operand {operand} for operator in. Should be a list."
                    )
                return models.FieldCondition(
                    key=key, match=models.MatchAny(any=operand)
                )
            elif operator == "nin":
                if not isinstance(operand, list):
                    raise ValueError(
                        f"Invalid operand {operand} for operator nin. Should be a list."
                    )
                return models.FieldCondition(
                    key=key, match=models.MatchExcept(**{"except": operand})
                )
            else:
                raise ValueError(
                    f"Invalid operator {operator} in filter:\n{json.dumps(filter, indent=4)}"
                )

        must = []
        should = []
        for key, value in filter.items():
            if key in ["_and", "_or"]:
                if not isinstance(value, list):
                    raise ValueError(
                        f"Invalid value {value} for operator {key}. Should be a list."
                    )
                for item in value:
                    if key == "_and":
                        must.append(process_filter(*list(item.items())[0]))
                    elif key == "_or":
                        should.append(process_filter(*list(item.items())[0]))
            else:
                must.append(process_filter(key, value))

        result = models.Filter(
            must=must if must else None,
            should=should if should else None,
        )

        return result


def file_name(path: str):
    return os.path.splitext(os.path.basename(path))[0]


def file_extension(path: str):
    return os.path.splitext(path)[1][1:]


def is_valid_date(date):
    try:
        date = str(date)
        parse(date)
        return True
    except ValueError:
        return False
