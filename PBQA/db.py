import json
import logging
import os
import uuid
from datetime import datetime
from time import time
from typing import List, Union

import yaml
from dateutil.parser import parse
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

log = logging.getLogger()


class DB:
    """A wrapper for the ChromaDB client with some added functionality for ease of use."""

    DEFAULT_ENCODER = "all-MiniLM-L6-v2"
    DEFAULT_RESULT_COUNT = 5

    def __init__(
        self,
        path: str,
        encoder: str = DEFAULT_ENCODER,
        reset: bool = False,
    ):
        """
        Initialize the DB client.

        Parameters:
        - path (str): The path to the database. Defaults to `paths.db_dir` from the config file.
        - encoder (str): The name of the SentenceTransformer model to use for encoding documents.
        - reset (bool, optional): Whether to reset the database. Defaults to False.
        """

        if not path:
            raise FileNotFoundError("No path provided for the database.")
        self.path = path

        self.client = QdrantClient(path=self.path)

        self.encoder = SentenceTransformer(encoder)

        if reset:
            log.info("Resetting db")
            self.reset()

        if "metadata" not in self.get_collections():
            self.metadata = self.client.create_collection(
                collection_name="metadata",
                vectors_config=models.VectorParams(
                    size=1, distance=models.Distance.COSINE
                ),
            )

        self.pattern_components = {}

    def load_pattern(self, path: str):
        if file_extension(path) not in ["yaml", "json"]:
            raise ValueError(
                "Invalid file type. Only .yaml and .json files are supported."
            )

        collection_name = file_name(path)
        if collection_name not in self.get_collections():
            self._add_from_file(path)

    def _add_from_file(
        self,
        path: str,
        **kwargs,
    ) -> models.CollectionInfo:
        """
        Add documents to a collection from a file.

        This method reads a file and adds the documents it contains to a collection. The file can be in either JSON or YAML format. The collection name is inferred from the file name, but can be overridden by passing a `collection_name` parameter.

        The file should contain a dictionary with at least a `data` key, which contains a list of dictionaries. Each dictionary in the `data` list represents a document, and should contain a `input` key with the document's input. Additional keys in the dictionary are treated as metadata for the document.

        Parameters:
        - path (str): The path to the file to read.

        Returns:
        - Topic: The collection that the documents were added to.
        """
        prev = time()

        collection_name = file_name(path)
        log.info(f"Populating collection {collection_name} with {path}")

        data = self.load_from_file(path)

        metadata = {k: v for k, v in data.items() if k != "examples"}
        if "system_prompt" not in metadata:
            log.warn(
                f"Pattern {collection_name} does not contain a system_prompt. This may lead to unexpected behavior. May alternatively be passed when calling the LLM."
            )

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
    ) -> str:
        """
        Get a collection by name.

        This method retrieves a collection by name. If the collection does not exist, it can be created by setting the `create` parameter to True.

        Parameters:
        - collection_name (str): The name of the collection to retrieve.
        - metadata (dict, optional): Metadata to pass when creating the collection. Defaults to {"hnsw:space": "l2"}.
        - **kwargs: Additional keyword arguments to pass when retrieving the collection.

        Returns:
        - Topic: The collection with the given name.
        """
        if collection_name not in self.get_collections():
            log.info(
                f"Creating collection {collection_name} with metadata:\n{yaml.dump(metadata, default_flow_style=False)}"
            )

            config = models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            )

            metadata["components"] = [
                key for key in metadata.keys() if key != "system_prompt"
            ]
            metadata["collection_name"] = collection_name
            self.client.upsert(
                collection_name="metadata",
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
            log.info(f"Collection {collection_name} already exists")

    def add(
        self,
        input: str,
        collection_name: str,
        time_added: time = time(),
        **kwargs,
    ) -> dict:
        """
        Add a document to a collection.

        This method adds a document to a collection. The document should contain an `input` key with the document's input. Additional keys in the dictionary are treated as metadata for the document.

        Parameters:
        - input (str): The document's input.
        - collection_name (str): The name of the collection to add the document to.
        - time_added (time, optional): The time the document was added. Defaults to the current time.
        - **kwargs: Additional keyword arguments to pass as metadata for the document.

        Returns:
        - dict: A dictionary representing the added document. The dictionary contains the document's input, id, and metadata.
        """

        def stringify_value(value):
            """
            Convert a value to its string representation.
            Dates are converted to a 'YYYY-MM-DD HH:MM:SS' format.
            Other types are converted using the str function.
            """
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%dT%H:%M:%S")
            elif isinstance(value, int) or isinstance(value, float):
                return value
            else:
                return str(value)

        # Use the stringify_value function to convert all kwargs values
        kwargs = {k: stringify_value(v) for k, v in kwargs.items()}

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
        collection_name: str,
    ) -> dict:
        """
        Get the metadata for a collection.

        This method retrieves the metadata for a collection.

        Parameters:
        - collection_name (str): The name of the collection to query.

        Returns:
        - dict: The metadata for the given collection.
        """

        if collection_name not in self.get_collections():
            raise ValueError(f"Collection {collection_name} not found")

        metadata = self.client.scroll(
            collection_name="metadata",
            scroll_filter=self.filter_to_qdrant_filter(
                {"collection_name": collection_name}
            ),
        )[0][0].payload

        return metadata

    def get_collections(self) -> List[str]:
        """Get the names of all collections in the database."""
        collections = self.client.get_collections().__dict__["collections"]

        if len(collections) == 0:
            return []

        return [collection.name for collection in collections]

    def n_collections(self):
        """Get the number of collections in the database."""
        try:
            return len(self.get_collections())
        except Exception as e:
            return 0

    def n(self, collection_name: str):
        """Get the number of documents in a collection."""
        if collection_name not in self.get_collections():
            raise ValueError(f"Collection {collection_name} not found")

        try:
            return self.client.count(collection_name)
        except Exception as e:
            return 0

    def reset(self):
        """Reset the database by deleting all collections."""
        for collection in [collection for collection in self.get_collections()]:
            self.client.delete_collection(collection_name=collection)

    def query(
        self,
        collection_name: str,
        input: str = "",
        n: int = DEFAULT_RESULT_COUNT,
        min_d: Union[int, None] = None,
        where: dict = None,
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

        - `$eq` - equal to (string, int, float)
        - `$ne` - not equal to (string, int, float)
        - `$gt` - greater than (int, float)
        - `$gte` - greater than or equal to (int, float)
        - `$lt` - less than (int, float)
        - `$lte` - less than or equal to (int, float)

        Note that metadata filters only search embeddings where the key exists. If a key is not present in the metadata, it will not be returned.

        To filter on document contents, supply a `where_document` filter dictionary to the query. We support two filtering keys: `$contains` and `$not_contains`.

        You can also use the logical operators `$and` and `$or` to combine multiple filters, and the inclusion operators `$in` and `$nin` to filter based on whether a value is in or not in a predefined list.

        Parameters:
        - collection_name (str): The name of the collection to query.
        - input (str): The input to match.
        - n (int, optional): The number of results to return. Defaults to DEFAULT_RESULT_COUNT.
        - min_d (Union[int, None], optional): The maximum distance for a document to be considered a match. If None, no maximum distance is used.
        - where (List[str], optional): A list of document content filters. Defaults to None.
        - **kwargs: Additional keyword arguments to pass as metadata filters.

        Returns:
        - List[dict]: A list of dictionaries, each representing a matching document. Each dictionary contains the document's input, id, distance from the query input, and metadata.
        """

        if collection_name not in self.get_collections():
            raise ValueError(
                f"Collection {collection_name} not found. Make sure to load the pattern first or create the collection manually."
            )

        if input == "":
            raise ValueError("Input cannot be empty. Use where method instead.")

        if n == -1:
            n = 20
        elif n < 1:
            return []

        query_filter = self.filter_to_qdrant_filter(where)

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
        collection_name: str,
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

        - `$eq` - equal to (string, int, float)
        - `$ne` - not equal to (string, int, float)
        - `$gt` - greater than (int, float)
        - `$gte` - greater than or equal to (int, float)
        - `$lt` - less than (int, float)
        - `$lte` - less than or equal to (int, float)

        Note that metadata filters only search embeddings where the key exists. If a key is not present in the metadata, it will not be returned.

        Parameters:
        - collection_name (str): The name of the collection to query.
        - n (int, optional): The number of results to return. Defaults to DEFAULT_RESULT_COUNT.
        - start (float, optional): The start time for the query. Defaults to None.
        - end (float, optional): The end time for the query. Defaults to None.
        - order_by (str, optional): The key to order the results by. Defaults to "time_added".
        - order_direction (str, optional): The direction to order the results in. Defaults to "desc".
        - **kwargs: Additional keyword arguments to pass as metadata filters.

        Returns:
        - List[dict]: A list of dictionaries, each representing a matching document. Each dictionary contains the document's input, id, and metadata.
        """

        if collection_name not in self.get_collections():
            raise ValueError(
                f"Collection {collection_name} not found. Make sure to load the pattern first or create the collection manually."
            )

        if n == -1:
            n = 20
        elif n < 1:
            return []

        if start:
            kwargs["time_added"] = {"$gte": start}
        if end:
            kwargs["time_added"] = {"$lte": end}

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
        def process_filter(key, value):
            if isinstance(value, dict):
                operator, operand = list(value.items())[0]
            else:
                operator, operand = "$eq", value

            if is_valid_date(operand):
                range = models.DatetimeRange()
            else:
                range = models.Range()

            if key == "$and":
                if not isinstance(operand, List):
                    raise ValueError(
                        f"Invalid operand {operand} for operator $and. Should be a list."
                    )
                return models.Filter(
                    must=[process_filter(*list(item.items())[0]) for item in operand]
                )
            elif key == "$or":
                if not isinstance(operand, List):
                    raise ValueError(
                        f"Invalid operand {operand} for operator $or. Should be a list."
                    )
                return models.Filter(
                    should=[process_filter(*list(item.items())[0]) for item in operand]
                )
            elif operator == "$eq":
                return models.FieldCondition(
                    key=key, match=models.MatchValue(value=operand)
                )
            elif operator == "$ne":
                return models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key=key, match=models.MatchValue(value=operand)
                        )
                    ]
                )
            elif operator == "$gt":
                range.gt = operand
                return models.FieldCondition(key=key, range=range)
            elif operator == "$gte":
                range.gte = operand
                return models.FieldCondition(key=key, range=range)
            elif operator == "$lt":
                range.lt = operand
                return models.FieldCondition(key=key, range=range)
            elif operator == "$lte":
                range.lte = operand
                return models.FieldCondition(key=key, range=range)
            elif operator == "$in":
                if not isinstance(operand, list):
                    raise ValueError(
                        f"Invalid operand {operand} for operator $in. Should be a list."
                    )
                return models.FieldCondition(
                    key=key, match=models.MatchAny(any=operand)
                )
            elif operator == "$nin":
                if not isinstance(operand, list):
                    raise ValueError(
                        f"Invalid operand {operand} for operator $nin. Should be a list."
                    )
                return models.FieldCondition(
                    key=key, match=models.MatchExcept(**{"except": operand})
                )
            else:
                raise ValueError(f"Invalid operator {operator}")

        if not filter:
            return None

        must = []
        should = []
        for key, value in filter.items():
            if key in ["$and", "$or"]:
                if not isinstance(value, list):
                    raise ValueError(
                        f"Invalid value {value} for operator {key}. Should be a list."
                    )
                for item in value:
                    if key == "$and":
                        must.append(process_filter(*list(item.items())[0]))
                    elif key == "$or":
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
