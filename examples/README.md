<h1 align="center">Examples and Explanations</h1>

# Patterns
Patterns are written in YAML and are used to define the structure of an LLM's response. The file must contain at least one key for every component of the response, each of which can have its own metadata or be left empty.

Below is an example of a pattern file for a weather query.

```yaml
system_prompt: Your job is to translate the user's input into a weather query. Reply with the json for the weather query and nothing else.
now:
  external: true
latitude:
  grammar: |
    root         ::= coordinate
    coordinate   ::= integer "." integer
    integer      ::= digit | digit digit
    digit        ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
longitude:
  grammar: ...
time:
  grammar: ...
examples:
- input: What will the weather be like tonight
  now: 2019-09-30 10:36
  latitude: 51.51
  longitude: 0.13
  time: 2019-09-30 20:00
- input: Could I see the stars tonight?
  ...
```

Components in a pattern file can have the following metadata:

- `external`: Specifies whether the component requires external data. If `true`, the data for the component must be provided in the `external` dictionary when querying the LLM.
- `grammar`: A [grammar](#grammar) that defines the structure of the component.

Besides components, a pattern file may also contain a system prompt. The system prompt is an optional instruction given to the LLM to guide its response. It may alternatively be passed as an argument to the `llm.ask()` method or left out entirely.

Lastly, examples may be provided in the pattern file for [multi-shot prompting](https://arxiv.org/abs/2005.14165) to improve the quality of the LLM's responses.

See the [conversation section](#conversation) for an example of a pattern with a system prompt, or the [function calling section](#function-calling) for an example of patterns with grammars, examples, and external data.

## Caching
Unless overridden, queries using the same pattern will use the same system prompt and base examples. This allows a large part of the response to be cached, speeding up generation. This can be disabled by setting `use_cache=False` in the `ask()` method.

To cache the patterns, PBQA will try to allocate a slot/process for each pattern-model pair in the llama.cpp server. As such, make sure to set `-np` to the number of unique combinations of patterns and models you want to enable caching for. 

Unless manually assigned, the slots are allocated in the order they are requested. If the number of available slots is exceeded, the last slot is reused for any excess pattern-model pairs. This ensures that the cache slot will never exceed the number of processes available.

The `assign_cache_slot` method can be used to manually assign a cache slot to a specific pattern-model pair. This method assigns a cache slot to a given pattern and model combination. Optionally, a specific cache slot can be provided, up to the number of available processes.


```python
from PBQA import DB, LLM


db = DB(path="examples/db")
db.load_pattern("examples/weather.yaml")

llm = LLM(db=db, host="127.0.0.1")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0,
)
llm.assign_cache_slot(pattern="weather", model="llama")
```

Note that, while usually the cache slot used for a query is the one assigned to the pattern-model pair, the cache slot can be overridden by passing the `cache_slot` parameter to the `llm.ask()` method.

# Conversation
Conversational agents are a common usecase for LLMs. While PBQA allows for [structured responses](#grammar), it can also be used to generate free-form responses. Below is an example of a pattern file for a conversational agent.

```yaml
system_prompt: You are a virtual assistant. Your tone is friendly and helpful.
reply:
```

Though no grammar or examples are provided, the response will still be steered by the system prompt.

## Setup
To use a pattern, first the vector database must be initialized and the necessary files loaded. After that, the LLM can be connected to the model(s) to be queried.

```py
from PBQA import DB, LLM


db = DB(path="examples/db")
db.load_pattern("examples/conversation.yaml")

llm = LLM(db=db, host="127.0.0.1")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
)
```

To create a conversational agent, only a single query is needed.

```py
while True:
    user_input = input("User\n> ")

    response = llm.ask(
        input=user_input,
        pattern="conversation",
        model="llama",
        n_hist=50,
    )

    db.add(
        input=user_input,
        collection_name="conversation",
        **response,
    )

    print(f"\nAssistant\n> {response['reply']}\n")
```

The `n_hist` parameter is used to specify the number of previous interactions to consider when generating a response. The history will always consist of `n_hist` entries in the database, ordered chronologically by the `time_added` field (optional argument in `db.add()`). This can be useful for maintaining context in a conversation.

After the response is generated, the exchange is added to the database for future reference.

By default, the database has a "collection" for each pattern that was loaded. And when unspecified, the default collection from which the history is retrieved when queried, is the pattern name. This can be overridden by specifying the `history_name` parameter in the `llm.ask()` method. Collections can be created with the `db.create_collection()` method.

_[Note.](#examples-and-history-caveat)_ The use of `n_hist` in conjunction with `n_examples` has not been properly tested yet. Using both parameters may lead to unexpected behavior.

# Function Calling
Another common usecase for LLMs is function calling. While PBQA doesn't call functions directly, using patterns, it is easy to create valid (json) objects to be used as input for tools. By combining patterns, it is possible to create an agent capable of navigating symbolic systems.

## Tool
The method `get_forecast` below uses the [Open-Meteo API](https://open-meteo.com/) to get a forecast for a given location and time. The function takes a latitude, longitude, and time as input and returns a forecast object.

```py
import requests


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

## Grammar
Each component in a pattern file can optionally have a grammar. Grammars are created to ensure that the LLM generates a valid response and provide it with some examples, as can be seen in [weather.yaml](weather.yaml). PBQA uses [GBNF grammars](https://github.com/ggerganov/llama.cpp/tree/master/grammars#gbnf-guide) to dictate the structure of a given pattern component. Take the `latitude` component for instance:

```gbnf
root         ::= coordinate
coordinate   ::= integer "." integer
integer      ::= digit | digit digit
digit        ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
```

This grammar ensures that the `latitude` component is a float with a single decimal point. The `longitude` and `time` components have similar grammars.

## Agent
To use patterns, first the vector database must be initialized and the necessary files loaded. After that, the LLM can be connected to the model(s) to be queried.

```py
from PBQA import DB, LLM


db = DB(path="examples/db")
db.load_pattern("examples/weather.yaml")
db.load_pattern("examples/answer_json.yaml")

llm = LLM(db=db, host="127.0.0.1")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0,
)
```

Creating a class akin to an 'Agent' may be useful for structuring sets of queries or patterns that are designed to accomplish specific tasks.

```py
from json import dumps
from time import strftime


class Agent:
    def __init__(self, db: DB, llm: LLM):
        self.db = db
        self.llm = llm

    def ask_weather(self, input: str):
        now = strftime("%Y-%m-%d %H:%M")
        weather_query = self.llm.ask(
            input=input,
            pattern="weather",
            model="llama",
            external={"now": now},
        )

        forecast = get_forecast(**weather_query)
        forecast = self._format_forecast(forecast)

        return self.llm.ask(
            input=input,
            pattern="answer_json",
            model="llama",
            external={"json": dumps(forecast)},
        )
```

Above, the `ask_weather` method is a tool for asking the LLM about a weather related question.

In the first query, the LLM is asked to provide a valid weather query, as defined by the [weather pattern](weather.yaml). This query object is then passed to the `get_forecast` function to obtain the forecast response from Open-Meteo. Lastly, the forecast is [formatted](#formatting) and turned into a json object, before being passed to the LLM to generate the final response.

## Inference
Running the function with the input:

    Could I see the stars tonight?

First results in the following query object:

```json
{
    "latitude": 51.51,
    "longitude": 0.13,
    "time": "2024-06-22 23:00"
}
```

Then, using the `get_forecast` [function](#tool), the following forecast object is generated:

```json
{
    "temperature": "19.3C",
    "precipitation_probability": "3%",
    "precipitation": "0.0mm",
    "cloud_cover": "64% coverage"
}
```

This forecast object is then [formatted](#formatting), dumped into a json string, and passed to the LLM to generate the final response:

```json
{
    "thought": "The temperature is quite cool at 19.3C, but the precipitation probability is low at 3%. However, the cloud cover is quite high at 64%, which might make it difficult to see the stars. It's not ideal conditions, but it's not impossible either.",
    "answer": "It might be a bit challenging to see the stars tonight due to the 64% cloud cover. However, the low precipitation probability and temperature are in your favor. If you're willing to brave the clouds, you might still be able to catch a glimpse of the stars."
}
```

As defined by the [answer_json.yaml](answer_json.yaml) pattern file, the LLM generates a response with both a `thought` and `answer` component. The preceding `thought` component allows the LLM to [think](https://arxiv.org/abs/2201.11903) about the provided data before giving a final answer to improve the quality of the response.

Regarding the weather query, note that the properties are based on both the input and the examples provided in the pattern file. While the input did not specify a location, the LLM defaulted to London's latitude and longitude. This is because the examples in the [pattern](weather.yaml) use the coordinates of London whenever no specific location is mentioned. As such, the LLM decided to do the same in this case.

This concept becomes more powerful when new examples are stored in the database, allowing the LLM to "[learn](#example-based-learning)" from past interactions and provide more accurate responses in the future.

By linking together patterns to interpret and manipulate data, an LLM can be used to dynamically set goals and navigate between tasks, or even break down problems recursively. This can be used to create (semi-)autonomous agents that can interact with other systems or users.

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

# Example Based Learning
The examples in the pattern file are included as part of every query to the LLM, unless `include_base_examples` is set to `False` in the `llm.ask()` method. Since caching is enabled by default, the increased prompt processing time for these examples only occurs once per pattern (per model). In addition to these base examples, more examples can also be added later for the LLM to learn from.

Take the following example:

```py
llm.ask(
    input="Is it going to rain tonight at home?",
    pattern="weather",
    model="llama",
    external={"now": strftime("%Y-%m-%d %H:%M")},
    n_example=1,
)
```

Based on the base examples in the pattern file, the LLM generates something akin to the following query object:

```json
{
    "latitude": 51.51,
    "longitude": 0.13,
    "time": "2024-06-22 21:00"
}
```

Ignoring the time component, the latitude and longitude are based on the examples in the pattern file, in this case representing London. Providing the LLM with another example  however, alters its response.

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
During a query, the examples retrieved from the database are formatted into messages between the user and the LLM. The user's input is always the first message consisting of the input and any external components. The LLM's response is the second message, consisting of the remaining components.

When either the user or the LLM has only one component, their message consists of only that component. If there are multiple components, their message is formatted into a json object.

The `n_examples` parameter is used to specify the number of examples that are provided to the LLM in addition to the base examples. The examples are always ordered by semantic similarity to the input query, with the most relevant example being the last one. Additional keyword arguments can be passed to the `llm.ask()` method to further [filter](#filtering) the examples.

### Feedback
Since most queries benefit from additional examples, setting up a system to process feedback can be an effective way to improve the LLM's responses over time. To that end, specific patterns can be created to classify and parse feedback, or even to have the LLM generate feedback for itself. After exporting, the feedback-improved examples can then be used to fine-tune the model.

### Examples and History Caveat
While using both `n_examples` and `n_hist` at once in a query should lead to a valid call, using both parameters is not recommended. The way the entries are currently formatted makes no distinction between examples and history, the history merely being appended after the examples. This may lead to unexpected behavior and poorer response quality. This is a known issue that will be addressed in a future update.

One way of solving this issue is to manually retrieve the history from the database and pass as an external component to the LLM. This way, the history can be formatted and filtered as needed before being passed to the LLM.

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
    input="Is it going to rain tonight at home?",
    pattern="weather",
    model="llama",
    external={"now": strftime("%Y-%m-%d %H:%M")},
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
    input="Is it going to rain tonight at home?",
    pattern="weather",
    model="llama",
    external={"now": strftime("%Y-%m-%d %H:%M")},
    n_example=1,
    feedback=True,  # or {"eq": True}
)
```

Now, the LLM will only receive examples tagged as feedback, which can be useful for providing specific examples to the LLM. Note that since `feedback` is not defined in the pattern file as a component, it will not be included in the response.

Besides queries to the LLM, filters are also used to retrieve examples from the database. The `db.query()` method is used to retrieve entries from the database based on the semantic similarity to the provided `input`. The `db.where()` method is used to retrieve entries based on the provided filters. Both methods use the same filtering syntax as the `llm.ask()` method, being passed as keyword arguments.

By default the `db.where()` method returns the entries in the order they were added to the database, with the most recent entry being the last one. Optionally, the `order_by` parameter can be used to specify a different field to order the entries by and the `order_direction` parameter to specify the order direction ("asc" or "desc"). When using a remote Qdrant server, the component to order the results by must first be indexed using the `db.index()` method. The `db.where()` method also supports the use of the `start` and `end` parameters to filter by time.