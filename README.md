<h1 align="center">Pattern Based Question and Answer</h1>

## Description
Pattern Based Question and Answer (PBQA) is a Python library that provides tools for querying LLMs and managing text embeddings. It combines [guided generation](examples/README.md#grammar) with [multi-shot prompting](https://arxiv.org/abs/2005.14165) to improve response quality and ensure consistency. By enforcing valid responses, PBQA makes it easy to combine the flexibility of LLMs with the reliability and control of symbolic approaches. 

 - [Installation](#installation)
 - [Usage](#usage)
 - [Patterns](#patterns)
 - [Nested Input Keys](#nested-input-keys)
 - [Cache](#cache)
 - [Roadmap](#roadmap)
 - [Relevant Literature](#relevant-literature)
 - [Contributing](#contributing)
 - [Support](#support)
 - [License](#license-and-acknowledgements)

## Installation
PBQA requires Python 3.12 or higher, and can be installed via pip:

```sh
pip install PBQA
```

Additionally, PBQA requires a running instance of llama.cpp to interact with LLMs. For instructions on installation, see the [llama.cpp repository](https://github.com/ggerganov/llama.cpp/tree/master?tab=readme-ov-file#usage).

## Usage
### llama.cpp
For instructions on hosting a model with llama.cpp, see the [following page](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#quick-start). Optionally, [caching](#cache) can be enabled to speed up generation.

### Python
PBQA provides a simple API for querying LLMs.

```python
from time import strftime
from pydantic import BaseModel
from PBQA import DB, LLM

# First, we define a schema for the weather query
class Weather(BaseModel):
    latitude: float
    longitude: float
    time: str

# Then, we set up a database at a specified path (or the host and port of a remote server)
db = DB(path="db")
# And define a pattern to use for generating responses
db.load_pattern(
    schema=Weather,
    examples="weather.yaml",
    system_prompt="Your job is to translate the user's input into a weather query object.",
    input_key="query",
)

# Next, we connect to the LLM server
llm = LLM(db=db, host="localhost")
# And connect to the model
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0,
)

# Finally, we query the LLM and receive a response based on the specified pattern
# Optionally, external data can be provided to the LLM which it can use in its response
weather_query = llm.ask(
    input={
        "query": "Could I see the stars tonight?",
        "now": "2024-09-30 10:36",
    },
    pattern="weather",
    model="llama",
)["response"]
```

Using the [weather.yaml](examples/weather.yaml) pattern file and [llama 3](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF) running on localhost:8080, the response should look something like this:

```json
{
    "latitude": 51.51,
    "longitude": 0.13,
    "time": "2024-09-30 23:00",
}
```

For more information, see the [examples](examples/README.md) directory.

### Patterns
Patterns are used to guide the LLM in generating responses. Each pattern needs at least a schema to define the expected output, and optionally a system prompt and example data. The system prompt is the main instruction given to the LLM telling it what to do. The example data is used to further guide the LLM in generating responses.

The example case above uses the Weather schema defined earlier in the code, a simple system prompt describing the task, and some sample data for the weather query.

While the example above uses an unmodified string to represent the time, it's also possible to use regex to restrict it further:

```py
class Weather(BaseModel):
    latitude: float
    longitude: float
    time: Annotated[
        str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}$")
    ]
```

Using the Field annotation and specifying a regex pattern, the LLM will only be able to generate responses that match the pattern. In this cae the LLM will only be able to generate responses that are in the format of a date and time in the format `YYYY-MM-DD HH:MM`.

Beyond the Pydantic schema, the user can also provide a system prompt and example data to help the LLM generate responses. Here is an excerpt from the [weather.yaml](examples/weather.yaml) file:


```yaml
- user:
    query: What will the weather be like tonight
    now: 2019-09-30 10:36
  assistant:
    latitude: 51.51
    longitude: -0.13
    time: 2019-09-30 20:00
- user:
    query: any idea if it'll be sunny tomorrow in Paris?
    now: 2016-11-02 12:15
  assistant:
    latitude: 48.86
    longitude: 2.35
    time: 2016-11-03 13:00
- user:
    query: will it be dry out by the time I get off work?
    now: 2025-06-12 09:23
  assistant:
    latitude: 51.51
    longitude: -0.13
    time: 2025-06-12 17:00
...
```

*Note.* While the assistant's response is validated against the schema when loaded, the user's input is free to be anything. This allows the user to provide any information they want (see [tool use](examples/tool_use.py)).

Any samples passed to the LLM through the `examples` parameter will be used as "base examples" for the pattern, which are examples that are loaded as part of every query to the LLM. Since caching is enabled by default, if your llama.cpp server is initialized correctly, the increased prompt processing time for these examples only occurs once per pattern (see [cache](#cache)). In addition to these base examples, more examples can also be added later for the LLM to learn from.

### Nested Input Keys
PBQA supports nested access to complex data structures through the `input_key` parameter. This enables working with hierarchical data like conversation histories, nested API responses, and complex application states.

#### Supported Syntax
- **Simple keys**: `"query"` - Direct dictionary access (backward compatible)
- **Dot notation**: `"user.query"` - Navigate nested dictionaries
- **Array indexing**: `"history[0]"` - Access array elements by index
- **Negative indexing**: `"history[-1]"` - Access array elements from the end
- **Combined paths**: `"user.history[0].input"` - Mix dots and array access

#### Examples
```python
# Simple key access (existing behavior)
db.load_pattern(
    schema=Response,
    input_key="query"
)

# Nested object access
db.load_pattern(
    schema=Response,
    input_key="user.query"
)

# Array indexing
db.load_pattern(
    schema=Response,
    input_key="messages[0]"
)

# Complex nested access
db.load_pattern(
    schema=Response,
    input_key="conversation.history[-1].content"
)
```

This feature enables PBQA to work seamlessly with complex conversation architectures and structured data formats while maintaining full backward compatibility with existing patterns. 

### Cache
Unless overridden, queries using the same pattern will use the same system prompt and base examples, allowing a large part of the response to be cached. This avoids the need reprocess those parts of the response, speeding up the query. This can be disabled by setting `use_cache=False` when invoking `llm.ask()`.

PBQA allocates a slot/process for each pattern-model pair in the llama.cpp server. Set `-np` to the number of unique combinations of patterns and models you want to enable caching for. Slots are allocated in the order they are requested, and if the number of available slots is exceeded, the last slot is reused for any excess pattern-model pairs.

You can manually assign a cache slot to a specific pattern-model pair using the `link` method. Optionally, a specific cache slot can be provided, up to the number of available processes. The cache slot used for a query can also be overridden by passing the `cache_slot` parameter to the `llm.ask()` method.

```py
from PBQA import DB, LLM


db = DB(path="db")
db.load_pattern(
    schema=Weather,
    examples="weather.yaml",
    system_prompt="Your job is to translate the user's input into a weather query object.",
    input_key="query",
)

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0,
)
llm.link(pattern="weather", model="llama")
```

Once a pattern-model pair is linked, the "model" parameter in the `ask()` method may also be omitted. The query will instead use the model assigned during the last appropriate `link` call.

## Roadmap
Future features in no particular order with no particular timeline:
 - [Reranking](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#post-reranking-rerank-documents-according-to-a-given-query)
 - Parallel query execution
 - Combining multi-shot prompting with message history
 - Multimodal support
 - Further speed improvements (possibly [batching](https://github.com/guidance-ai/guidance?tab=readme-ov-file#guidance-acceleration))
 - Support for more LLM backends

## Relevant Literature
 - [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
 - [Many-Shot In-Context Learning (Aragwal, 2024)](https://arxiv.org/abs/2404.11018)
 - [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
 - [Using Grammar Masking to Ensure Syntactic Validity in LLM-based Modeling Tasks (Lukas et al., 2024)](https://arxiv.org/abs/2407.06146)

## Contributing
Contributions are welcome! If you have any suggestions or would like to contribute, please open an issue or a pull request.

## Support
If you want to support the development of PBQA, consider [buying me a coffee](https://ko-fi.com/baagsma). Any support is greatly appreciated!

## License and Acknowledgements
This project is licensed under the terms of the MIT License. For more details, see the [LICENSE file](./LICENSE).

[Qdrant](https://github.com/qdrant/qdrant-client) is a vector database that provides an API for managing and querying text embeddings. PBQA uses Qdrant to store and retrieve text embeddings.

[llama.cpp](https://github.com/ggerganov/llama.cpp) is a C++ library that provides an easy-to-use interface for running LLMs on a wide variety of hardware. It includes support for Apple silicon, x86 architectures, and NVIDIA GPUs, as well as custom CUDA kernels for running LLMs on AMD GPUs via HIP. PBQA uses llama.cpp to interact with LLMs.

[Pydantic](https://github.com/pydantic/pydantic) is a Python library that provides a powerful and flexible way to define data models.

PBQA was originally developed by Bart Haagsma as part of different project. If you have any questions or suggestions, please feel free to contact me at dev.baagsma@gmail.com.
