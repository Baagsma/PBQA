import json
import logging
import os
import uuid
from datetime import datetime
from time import time
from typing import List, Union
import requests

import yaml
from dateutil.parser import parse
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from hashlib import sha256

log = logging.getLogger()


class DB:
    """A wrapper for the Qdrant client with some added functionality for ease of use."""

    DEFAULT_ENCODER = "all-MiniLM-L6-v2"
    DEFAULT_RESULT_COUNT = 5
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
        - auto_update (bool, optional): Whether to automatically update the database when the patterns change. Defaults to False.
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
        self, path: str, pattern_name: str = None, collection_name: str = None
    ):
        """
        Load a pattern from a file.

        This method loads a pattern from a file and adds it to the database. The file can be in either JSON or YAML format. The pattern name is inferred from the file name, but can be overridden by passing a `pattern_name` parameter.

        The file must contain at least one key for every component of the response, each of which can have its own metadata or be left empty. The collection name is inferred from the pattern name, but can be overridden by passing a `collection_name` parameter.

        Parameters:
        - path (str): The path to the file to read.
        - pattern_name (str, optional): The name of the pattern to load. Defaults to None.
        - collection_name (str, optional): The name of the collection to load the pattern into. Defaults to None.
        """

        if file_extension(path) not in ["yaml", "json"]:
            raise ValueError(
                "Invalid file type. Only .yaml and .json files are supported."
            )

        data = self.load_from_file(path)
        pattern_name = pattern_name or file_name(path)
        collection_name = collection_name or pattern_name

        if collection_name not in self.get_collections():
            self._add_from_file(path, collection_name=collection_name)
            self.index(collection_name, "time_added", "float")
            return

        current_hash = sha256(json.dumps(data).encode()).hexdigest()
        stored_hash = self.get_metadata(collection_name)["hash"]

        if current_hash == stored_hash:
            log.info(
                f'Pattern "{pattern_name}" already loaded into collection "{collection_name}"'
            )
            return

        if self.auto_update:
            log.info(
                f'Pattern "{pattern_name}" was updated. Updating collection "{collection_name}"'
            )
            self.delete_collection(collection_name)
            self._add_from_file(path, collection_name=collection_name)
            self.index(collection_name, "time_added", "float")
        else:
            log.warn(
                f'Pattern "{pattern_name}" was updated. To update the collection, set auto_update=True when initializing the DB or manually call db.delete_collection("{collection_name}") and db.load_pattern("{path}")'
            )

    def _add_from_file(
        self,
        path: str,
        collection_name: str,
        **kwargs,
    ) -> str:
        """
        Add documents to a collection from a file.

        This method reads a file and adds the documents it contains to a collection. The file can be in either JSON or YAML format. The collection name is inferred from the file name, but can be overridden by passing a `collection_name` parameter.

        The file should contain a dictionary with at least a `data` key, which contains a list of dictionaries. Each dictionary in the `data` list represents a document, and should contain a `input` key with the document's input. Additional keys in the dictionary are treated as metadata for the document.

        Parameters:
        - path (str): The path to the file to read.
        - collection_name (str): The name of the collection to add the documents to.

        Returns:
        - str: The collection that the documents were added to.
        """
        prev = time()

        log.info(f'Populating collection "{collection_name}" with {path}')

        data = self.load_from_file(path)

        metadata = {k: v for k, v in data.items() if k != "examples"}

        if "system_prompt" not in metadata:
            log.warn(
                f"Pattern {collection_name} does not contain a system_prompt. This may lead to unexpected behavior. May alternatively be passed when calling the LLM."
            )

        metadata["components"] = [
            key for key in metadata.keys() if key != "system_prompt"
        ]
        metadata["pattern_name"] = file_name(path)
        metadata["hash"] = sha256(json.dumps(data).encode()).hexdigest()

        self.create_collection(
            collection_name=collection_name,
            metadata=metadata,
            **kwargs,
        )

        if "examples" in data:
            for item in data["examples"]:
                self.add(
                    **item,
                    collection_name=collection_name,
                    base_example=True,
                )

            log.info(
                f"Added {len(data['examples'])} documents in {(time() - prev):.1f} s"
            )

        return collection_name

    def create_collection(
        self,
        collection_name: str,
        metadata: dict = {},
        **kwargs,
    ):
        """
        Get a collection by name.

        This method retrieves a collection by name. If the collection does not exist, it can be created by setting the `create` parameter to True.

        Parameters:
        - collection_name (str): The name of the collection to retrieve.
        - metadata (dict, optional): Metadata to pass when creating the collection. Defaults to {"hnsw:space": "l2"}.
        - **kwargs: Additional keyword arguments to pass when retrieving the collection.
        """
        if collection_name not in self.get_collections():
            log.info(
                f'Creating collection "{collection_name}" with metadata:\n{yaml.dump(metadata, default_flow_style=False)}'
            )

            config = models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
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
        else:
            log.info(f'collection "{collection_name}" already exists')

    def delete_collection(self, collection_name: str):
        if collection_name not in self.get_collections():
            raise ValueError(f'Collection "{collection_name}" not found')

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
        input: str,
        collection_name: str,
        time_added: float = None,
        **kwargs,
    ) -> dict:
        """
        Add a document to a collection.

        This method adds a document to a collection. The document should contain an `input` key with the document's input. Additional keys in the dictionary are treated as metadata for the document.

        Parameters:
        - input (str): The document's input.
        - collection_name (str): The name of the collection to add the document to.
        - time_added (float, optional): The time the document was added in Unix time. Defaults to time().
        - **kwargs: Additional keyword arguments to pass as metadata for the document.

        Returns:
        - dict: A dictionary representing the added document. The dictionary contains the document's input, id, and metadata.
        """
        if collection_name not in self.get_collections():
            raise ValueError(
                f'collection "{collection_name}" not found. Make sure to load the pattern first or create the collection manually.'
            )

        time_added = time_added or time()

        doc_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector=self.encoder.encode(input),
                    payload={
                        **kwargs,
                        "input": input,
                        "time_added": time_added,
                    },
                )
            ],
        )

        return {"input": input, "id": doc_id, **kwargs}

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

        if not pattern and not collection_name:
            raise ValueError("Either pattern or collection_name must be provided.")

        if not (
            pattern in self.get_patterns() or collection_name in self.get_collections()
        ):
            raise ValueError(
                f'Neither pattern "{pattern}" nor collection "{collection_name}" found. Make sure to load the pattern first or create the collection manually.'
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
                f"Multiple metadata entries found for {'pattern' if pattern else 'collection'} {pattern if pattern else collection_name}. This is not supported."
            )

        return metadata[0].payload

    def get_collections(self) -> List[str]:
        """Get the names of all collections in the database."""
        collections = self.client.scroll(
            collection_name=self.metadata_collection_name,
            limit=500,
        )[0]
        if len(collections) == 0:
            return []
        return [collection.payload["collection_name"] for collection in collections]

    def get_collection(self, pattern: str) -> str:
        """Get the name of the collection for a given pattern."""
        return self.get_metadata(pattern)["collection_name"]

    def get_patterns(self) -> List[str]:
        """Get the names of all patterns in the database."""
        collections = self.client.scroll(
            collection_name=self.metadata_collection_name,
            limit=500,
        )[0]
        if len(collections) == 0:
            return []
        return [collection.payload["pattern_name"] for collection in collections]

    def n_collections(self):
        """Get the number of collections in the database."""
        try:
            return len(self.get_collections())
        except Exception as e:
            return 0

    def n(self, collection_name: str):
        """Get the number of documents in a collection."""
        if collection_name not in self.get_collections():
            raise ValueError(f'collection "{collection_name}" not found')

        try:
            return self.client.count(collection_name)
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

        log.warn("Resetting database")

        for collection in self.get_collections():
            self.client.delete_collection(collection)
        self.client.delete_collection(self.metadata_collection_name)

    def query(
        self,
        pattern: str,
        input: str = "",
        n: int = DEFAULT_RESULT_COUNT,
        min_d: Union[int, None] = None,
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
        - pattern (str): The name of the pattern to query.
        - input (str): The input to match.
        - n (int, optional): The number of results to return. Defaults to DEFAULT_RESULT_COUNT.
        - min_d (Union[int, None], optional): The maximum distance for a document to be considered a match. If None, no maximum distance is used.
        - where (dict, optional): A dictionary of metadata filters to apply. Defaults to None.

        Returns:
        - List[dict]: A list of dictionaries, each representing a matching document. Each dictionary contains the document's input, id, distance from the query input, and metadata.
        """

        if pattern not in self.get_patterns():
            if pattern not in self.get_collections():
                raise ValueError(
                    f'Neither pattern "{pattern}" nor collection "{collection_name}" found. Make sure to load the pattern first or create the collection manually.'
                )
            else:
                log.warn(
                    f'No collection associated with pattern "{pattern}". Using collection name "{pattern}" for query instead.'
                )
                collection_name = pattern
        else:
            collection_name = self.get_collection(pattern)

        if input == "":
            raise ValueError("Input cannot be empty. Use where method instead.")

        if n == -1:
            n = 20
        elif n < 1:
            return []

        query_filter = self.filter_to_qdrant_filter(kwargs)

        docs = self.client.search(
            collection_name=collection_name,
            query_filter=query_filter,
            limit=n,
            query_vector=self.encoder.encode(input),
            score_threshold=min_d,
        )

        results = [
            {
                "input": doc.payload["input"],
                "id": doc.id,
                **doc.payload,
            }
            for doc in docs
        ]

        return results

    def where(
        self,
        pattern: str,
        n: int = DEFAULT_RESULT_COUNT,
        start: float = None,
        end: float = None,
        order_by: str = "time_added",
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
        - pattern (str): The name of the pattern to query.
        - n (int, optional): The number of results to return. Defaults to DEFAULT_RESULT_COUNT.
        - start (float, optional): The start time for the query. Defaults to None.
        - end (float, optional): The end time for the query. Defaults to None.
        - order_by (str, optional): The key to order the results by. Defaults to "time_added".
        - order_direction (str, optional): The direction to order the results in. Must be "asc" or "desc". Defaults to "desc".
        - **kwargs: Additional keyword arguments to pass as metadata filters.

        Returns:
        - List[dict]: A list of dictionaries, each representing a matching document. Each dictionary contains the document's input, id, and metadata.
        """

        if pattern not in self.get_patterns():
            if pattern not in self.get_collections():
                raise ValueError(
                    f'Neither pattern "{pattern}" nor collection "{collection_name}" found. Make sure to load the pattern first or create the collection manually.'
                )
            else:
                log.warn(
                    f'No collection associated with pattern "{pattern}". Using collection name "{pattern}" for query instead.'
                )
                collection_name = pattern
        else:
            collection_name = self.get_collection(pattern)

        if n == -1:
            n = 20
        elif n < 1:
            return []

        if start:
            kwargs["time_added"] = {"gte": start}
        if end:
            kwargs["time_added"] = {"lte": end}

        scroll_filter = self.filter_to_qdrant_filter(kwargs)

        docs = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=n,
            order_by=models.OrderBy(
                key=order_by,
                direction=order_direction,
            ),
        )

        results = [
            {
                "input": doc.payload["input"],
                "id": doc.id,
                **doc.payload,
            }
            for doc in docs[0]
        ]

        return results

    def index(self, pattern: str, component: str, type: str):
        """
        Index a pattern component for filtering and ordering.

        This method indexes a pattern component for filtering and ordering (https://qdrant.tech/documentation/concepts/indexing/#payload-index).

        Parameters:
        - pattern (str): The name of the pattern to index.
        - component (str): The name of the component to index.
        - type (str): The type of index to create. Can be "keyword", "integer", "float", "bool", "geo", "datetime", or "text".
        """
        if not self.use_remote:
            return

        if pattern not in self.get_collections():
            raise ValueError(f'Pattern "{pattern}" not found')
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
            collection_name=pattern, field_name=component, field_type=type
        )
        log.info(f"Indexed {component} in {pattern} as {type}")

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
                "Filetype '{filetype}' not supported for {file}".format(
                    filetype=os.path.splitext(path)[1][1:],
                    file=path,
                )
            )

    @staticmethod
    def filter_to_qdrant_filter(filter: dict) -> models.Filter:
        if not filter or not filter.keys():
            return None

        log.info(f"Filter:\n{yaml.dump(filter, default_flow_style=False)}")

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
                return models.FieldCondition(
                    key=key, match=models.MatchValue(value=operand)
                )
            elif operator == "ne":
                return models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key=key, match=models.MatchValue(value=operand)
                        )
                    ]
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
                raise ValueError(f"Invalid operator {operator}")

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
