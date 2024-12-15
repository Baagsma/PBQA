<h1 align="center">Examples and Explanations</h1>

## Patterns
Patterns are written using Pydantic models and optional YAML examples to define the structure of an LLM's response. Each pattern requires at least a schema to define the expected output format, and can optionally include a system prompt and example data to guide the LLM's responses through multi-shot prompting.

Below is an example of a pattern setup for a weather query:

```python
from typing import Annotated
from pydantic import BaseModel, Field
from PBQA import DB, LLM


class Weather(BaseModel):
    latitude: float
    longitude: float
    time: Annotated[
        str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}$")
    ]

db = DB(path="db")
db.load_pattern(
    schema=Weather,
    examples="weather.yaml",
    system_prompt="Your job is to translate the user's input into a weather query object.",
    input_key="query"
)
```

When loading a pattern, several components can be specified:
- schema: A Pydantic model that defines the structure and validation rules for the response
- examples: Path to a YAML file containing example interactions for multi-shot prompting
- system_prompt: An optional instruction given to the LLM to guide its response
- input_key: The key used to embed the example in the database (defaults to "input")

The examples can be provided in YAML format to demonstrate desired behavior:
```yaml
- user:
    query: "What will the weather be like tonight?"
    now: "2019-09-30 10:36" 
  assistant:
    latitude: 51.51
    longitude: -0.13
    time: "2019-09-30 20:00"
```

Besides the schema, having a well-crafted system prompt and relevant examples is helpful in guiding the LLM to generate appropriate responses. The system prompt provides the overall context and instructions, while examples help demonstrate the expected input-output patterns.

## Regex Patterns
Using `Field(pattern=r"...")` its possible to define a regex pattern for a specific field in the schema. This can be useful for validating the input and ensuring it matches the expected format. To use this, make sure to import `Annotated` from `typing` and `Field` from `pydantic`. Also note that the field must be a string and that the pattern starts with `^` and ends with `$`.

## Conversation
Conversational agents are a common use case for LLMs. While PBQA allows for structured responses through Pydantic models, it can also be used for free-form conversation. Below is an example of a simple conversational pattern:

```python
class Conversation(BaseModel):
    reply: str

db = DB(path="db")
db.load_pattern(
    schema=Conversation,
    system_prompt="You are a virtual assistant. Your tone is friendly and helpful."
)
```

### Setup
To use a pattern, first the vector database must be initialized and the necessary patterns loaded. After that, the LLM can be connected to the model(s) to be queried:

```python
llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
)
```

To create a conversational agent, implement a loop for continuous interaction:

```python
while True:
    user_input = input("User\n> ")

    response = llm.ask(
        input=user_input,
        pattern="conversation",
        model="llama",
        n_hist=50,  # number of previous exchanges to consider
    )["response"]

    db.add(
        input=user_input,
        collection_name="conversation",
        **response,
    )

    print(f"\nAssistant\n> {response['reply']}\n")
```

The `n_hist` parameter specifies the number of previous interactions to consider when generating a response. The history will always consist of `n_hist` entries in the database, ordered chronologically by the `time_added` field. This helps maintain context in longer conversations.

After each response is generated, the exchange is added to the database for future reference. By default, the database has a "collection" for each pattern that was loaded. When unspecified, the default collection from which the history is retrieved is the pattern name. This can be overridden using the `history_name` parameter in `llm.ask()`.

## Function Calling
Another common use case for LLMs is function calling. While PBQA doesn't call functions directly, using patterns it's easy to create valid structured objects to be used as input for tools. By combining patterns, it's possible to create an agent capable of navigating between different tasks and handling complex interactions.

### Tool
Here's an example tool that uses the Open-Meteo API to get weather forecasts:

```python
def get_forecast(
    latitude: float,
    longitude: float,
    time: str,
):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,precipitation_probability,precipitation,cloud_cover&timeformat=unixtime"
    response = requests.get(url)
    data = response.json()["hourly"]

    target_epoch = datetime.fromisoformat(time).timestamp()
    closest_index = min(
        range(len(data["time"])),
        key=lambda i: abs(data["time"][i] - target_epoch),
    )

    return {
        "temperature": data["temperature_2m"][closest_index],
        "precipitation_probability": data["precipitation_probability"][closest_index],
        "precipitation": data["precipitation"][closest_index],
        "cloud_cover": data["cloud_cover"][closest_index],
    }
```

### Agent
Creating a class akin to an 'Agent' can be useful for structuring sets of queries or patterns designed to accomplish specific tasks:

```python
class ThinkAndAnswer(BaseModel):
    thought: str
    answer: str

class Agent:
    def __init__(self, db: DB, llm: LLM):
        self.db = db
        self.llm = llm

    def ask_weather(self, input: str):
        # Get structured weather query from user input
        weather_query = self.llm.ask(
            input={"query": input, "now": strftime("%Y-%m-%d %H:%M")},
            pattern="weather",
            model="llama",
        )["response"]

        # Get actual forecast data
        forecast = get_forecast(**weather_query)
        forecast = self._format_forecast(forecast)

        # Generate natural language response
        return self.llm.ask(
            input={"query": input, "json": json.dumps(forecast)},
            pattern="thinkandanswer",
            model="llama",
        )

    def _format_forecast(self, forecast: dict):
        return {
            "temperature": f'{forecast["temperature"]}C',
            "precipitation_probability": f'{forecast["precipitation_probability"]}%',
            "precipitation": f'{forecast["precipitation"]}mm',
            "cloud_cover": f'{forecast["cloud_cover"]}% coverage'
        }
```

### Inference
Running the agent with the input "Could I see the stars tonight?" demonstrates how patterns can be chained together:

1. First, the weather query pattern generates structured parameters:
    ```json
    {
        "latitude": 51.51,
        "longitude": -0.13,
        "time": "2024-06-22 23:00"
    }
    ```
2. These parameters are used to get forecast data, which is formatted:
    ```json
    {
        "temperature": "19.3C",
        "precipitation_probability": "3%",
        "precipitation": "0.0mm",
        "cloud_cover": "64% coverage"
    }
    ```
3. Finally, the ThinkAndAnswer pattern generates a natural language response:

    ```json
    {
        "thought": "The temperature is quite cool at 19.3C, but the precipitation probability is low at 3%. However, the cloud cover is quite high at 64%, which might make it difficult to see the stars. It's not ideal conditions, but it's not impossible either.",
        "answer": "It might be a bit challenging to see the stars tonight due to the 64% cloud cover. However, the low precipitation probability and temperature are in your favor. If you're willing to brave the clouds, you might still be able to catch a glimpse of the stars."
    }
    ```

As defined by the schema, the LLM generates a response with both a `thought` and `answer` component. The preceding `thought` component allows the LLM to [think](https://arxiv.org/abs/2201.11903) about the provided data before giving a final answer to improve the quality of the response.

Regarding the weather query, note that the properties are based on both the input and the provided examples. While the input did not specify a location, the LLM defaulted to London's latitude and longitude. This is because the [examples](weather.yaml) use the coordinates of London whenever no specific location is mentioned. As such, the LLM decided to do the same in this case.

This concept becomes more powerful when new examples are stored in the database, allowing the LLM to "[learn](#example-based-learning)" from past interactions and provide more accurate responses in the future.

### Formatting
Before passing the forecast object to the LLM, the data from the Open-Meteo API is formatted into a more LLM-friendly json object. This is done to prevent misinterpretation of the data by the LLM as much as possible.

```json
{
    "temperature": "19.3C",
    "precipitation_probability": "3%",
    "precipitation": "0.0mm",
    "cloud_cover": "64% coverage"
}
```

In this case, the unit of measurement is added to the temperature and precipitation values, a percentage sign is added to the precipitation probability, and the cloud cover is explicitly expressed as a coverage percentage (as opposed to probability). Each model will have its own preferences for how data is formatted, down to whitespaces and punctuation. As such, it may be valuable to test different formatting strategies to see which one works best for a given model.

## Example Based Learning
The examples passed in the `db.load_pattern()` method are included as part of every query to the LLM, unless `include_base_examples` is set to `False` in the `llm.ask()` method. Since caching is enabled by default, the increased prompt processing time for these examples only occurs once per pattern (per model). In addition to these base examples, more examples can also be added later for the LLM to learn from.

Take the following example:

```py
llm.ask(
    input={
        "query": "Is it going to rain tonight at home?",
        "now": strftime("%Y-%m-%d %H:%M")
    },
    pattern="weather",
    model="llama",
    n_example=1,
)
```

Based on the base examples, the LLM generates something akin to the following query object:

```json
{
    "latitude": 51.51,
    "longitude": 0.13,
    "time": "2024-06-22 21:00"
}
```

Ignoring the time component, the latitude and longitude are based on the loaded examples, in this case representing London. Providing the LLM with another example  however, alters its response.

```py
db.add(
    input="What's the weather like at home?",
    collection_name="weather",
    latitude=48.86,
    longitude=2.35,
    now="2021-03-02 14:46",
    time="2021-03-02 14:46",
)
```

When the LLM is now queried with the same input as before, its answer is different:

```json
{
    "latitude": 48.86,
    "longitude": 2.35,
    "time": "2024-06-22 22:00"
}
```

Based on the new example, the LLM has learned to associate "home" with Paris. Though this is a simple example, the same principle can be applied to more complex queries and patterns. By providing the LLM with more examples, it can learn to generate more accurate responses over time.

### Repetition Caveat
In the example above, the question in the example was different from the one in the query. This is since, in the current implementation, the most relevant example is the last example provided in the prompt. The repetition penalty of an LLM biases it against repeating the same text twice. This leads the LLM to avoid answering the query in the same way as the example, causing it to generate a different response. Sadly, the problem cannot be solved by simply lowering the `repeat_penalty`, though it can be remedied by providing more examples to the LLM (usually two is sufficient).

This is a known issue that will be addressed a future update.

## Examples
During a query, the examples retrieved from the database are formatted into messages between the user and the LLM. The user's input is always the first message, taking the form of either a string or a dictionary with a key named `input` (or the key specified in the `input_key` parameter). The LLM's response is the second message, consisting of the remaining components.

For a message to get passed as a string without being parsed as a dictionary, either the input must be a string on the user's side, or the schema must consist of a single string component on the LLM's side.

The `n_examples` parameter is used to specify the number of examples that are provided to the LLM in addition to the base examples. The examples are always ordered by semantic similarity to the input query, with the most relevant example being the last one. Additional keyword arguments can be passed to the `llm.ask()` method to further [filter](#filtering) the examples.

### Feedback
Since most queries benefit from additional examples, setting up a system to process feedback can be an effective way to improve the LLM's responses over time. To that end, specific patterns can be created to classify and parse feedback, or even to have the LLM generate feedback for itself. After exporting, the feedback-improved examples can then be used to fine-tune the model.

### Examples and History Caveat
While using both `n_examples` and `n_hist` at once in a query should lead to a valid call, using both parameters is not recommended. The way the entries are currently formatted makes no distinction between examples and history, the history merely being appended after the examples. This may lead to unexpected behavior and poorer response quality. This is a known issue that will be addressed in a future update.

One way of solving this issue is to manually retrieve the history from the database and pass as it to the LLM as part of the input. This way, the history can be formatted and filtered as needed before being passed to the LLM.

## Filtering
Any additional keyword arguments passed to the `llm.ask()` method are used to filter the examples provided to the LLM. 

To filter on metadata, supply the metadata field and its desired value directly as keyword arguments. For more complex queries, you can use a dictionary with the following structure:

```py
{
    operator: value
}
```

Metadata filtering supports the following operators:

- `eq` - equal to (string, int, float)
- `ne` - not equal to (string, int, float)
- `gt` - greater than (int, float)
- `gte` - greater than or equal to (int, float)
- `lt` - less than (int, float)
- `lte` - less than or equal to (int, float)

Note that metadata filters only search embeddings where the key exists. If a key is not present in the metadata, it will not be returned.

If no operator is specified, the default is `eq`, allowing for simple equality checks. Alternatively, `_and` and `_or` can be used and nested to create more complex queries.

```py
llm.ask(
    input={
        "query": "Is it going to rain tonight at home?",
        "now": strftime("%Y-%m-%d %H:%M")
    }
    pattern="weather",
    model="llama",
    n_example=1,
    _or=[
        {"feedback": True},
        {"latitude": {"lt": 50}},
    ],
)
```

An example usecase is to filter for examples specifically tagged as feedback. This can be done by adding a `feedback` key to a given example before adding it to the database.

```py
db.add(
    input="What's the weather like at home?",
    collection_name="weather",
    latitude=48.86,
    longitude=2.35,
    now="2021-03-02 14:46",
    time="2021-03-02 14:46",
    feedback=True,
)
```

Then, when querying the LLM, the examples can be filtered for feedback examples only.

```py
llm.ask(
    input={
        "query": "Is it going to rain tonight at home?",
        "now": strftime("%Y-%m-%d %H:%M")
    }
    pattern="weather",
    model="llama",
    n_example=1,
    feedback=True,  # or {"eq": True}
)
```

Now, the LLM will only receive examples tagged as feedback, which can be useful for providing specific examples to the LLM. Note that since `feedback` is not defined in the schema as a component, it will not be included in the response.

Besides queries to the LLM, filters are also used to retrieve examples from the database. The `db.query()` method is used to retrieve entries from the database based on the semantic similarity to the provided `input`. The `db.where()` method is used to retrieve entries based on the provided filters. Both methods use the same filtering syntax as the `llm.ask()` method, being passed as keyword arguments.

By default the `db.where()` method returns the entries in the order they were added to the database, with the most recent entry being the last one. Optionally, the `order_by` parameter can be used to specify a different field to order the entries by and the `order_direction` parameter to specify the order direction ("asc" or "desc"). When using a remote Qdrant server, the component to order the results by must first be indexed using the `db.index()` method. The `db.where()` method also supports the use of the `start` and `end` parameters to filter by time.

## Reranking
Reranking is a technique that involves a [dedicated model](https://huggingface.co/BAAI/bge-reranker-v2-m3) to rank any provided documents based on their relevance to input. While embeddings can be used to retrieve documents that are semantically similar, reranking can be used for more complex queries at the cost of increased latency. It is recommended to first retrieve relevant documents before ranking them to get the best of both worlds.

Any models that support reranking will be automatically recognized when connected through the `llm.connect_model()` method. This requires `--rerank --pooling rank` to be passed when starting up a llama.cpp server. After connecting to a model, the `llm.rerank()` method can be used to rerank a query using the reranking model.

```python
from PBQA import DB, LLM

db = DB(path="db")
llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="rerank",
    port=8080,
)

options = [
    "Search tool: The search tool is able to answer questions about current events, historical events, companies, products, and more.",
    "Weather tool: The weather tool is able to provide information on temperature, humidity, wind speed, and precipitation.",
    "Math tool: The math tool is able to perform basic arithmetic operations such as addition, subtraction, multiplication, and division.",
]

result = llm.rerank(
    "What's 10 + 20?",
    "rerank",
    documents=options,
    n=1,
)[0]
```

The example above should return the 'Math tool' as the first result, since it is the most relevant to the input. [See for more info](https://github.com/ggerganov/llama.cpp/pull/9510).

## Caching
Unless overridden, queries using the same pattern will use the same system prompt and base examples. This allows a large part of the response to be cached, speeding up generation. This can be disabled by setting use_cache=False in the ask() method.

To cache the patterns, PBQA will try to allocate a slot/process for each pattern-model pair in the llama.cpp server. As such, make sure to set -np to the number of unique combinations of patterns and models you want to enable caching for.

Unless manually assigned, the slots are allocated in the order they are requested. If the number of available slots is exceeded, the last slot is reused for any excess pattern-model pairs. This ensures that the cache slot will never exceed the number of processes available.

The link method can be used to manually assign a cache slot to a specific pattern-model pair:

```python
from PBQA import DB, LLM

db = DB(path="db")
db.load_pattern(
    schema=Weather,
    examples="weather.yaml",
    system_prompt="Your job is to translate the user's input into a weather query object.",
    input_key="query"
)

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0
)
llm.link(pattern="weather", model="llama")
```

### Store Cache
To store the cache, the `--slot-save-path` flag must be passed to the llama.cpp server. This will enable the server to save the cache to disk. By default, upon connecting to a model, PBQA will attempt to connect to the slot saving endpoint and check if it is available, enabling the cache saving by default. This feature can be disabled by setting `store_cache=False` in the `llm.connect_model()` method.

Using cache saving effectively circumvents the need to us multiple processes for the each pattern-model pair. When using the `llm.ask()` method, the cache will be loaded into the linked slot automatically. Enabling cache saving significantly speeds up the generation of responses by preventing reprocessing of the same prompts, even between processes.

Note that while this frees up more memory, it comes at the cost of disk space.