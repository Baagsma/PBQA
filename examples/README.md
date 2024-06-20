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

- `external`: Specifies whether the component requires external data. If `true`, the component will be passed as an argument to the `llm.ask()` method.
- `grammar`: A [grammar](#grammar) that defines the structure of the component.

Besides components, a pattern file may also contain a system prompt. The system prompt is an optional instruction given to the LLM to guide its response. It may alternatively be passed as an argument to the `llm.ask()` method, or left out entirely.

Lastly, examples may be provided in the pattern file for [multi-shot prompting](https://arxiv.org/abs/2005.14165) to improve the quality of the LLM's responses.

See the [conversation section](#conversation) for an example of a pattern with a system prompt, or the [function calling section](#function-calling) for an example of patterns with grammars, examples, and external data.

# Conversation
Conversational agents are a common usecase for LLMs. While PBQA allows for [structured responses](#grammar), it can also be used to generate free-form responses. Below is an example of a pattern file for a conversational agent.

```yaml
system_prompt: You are a virtual assistant. Your tone is friendly and helpful.
reply:
```

Though no grammar or examples are provided, the response will still be steered by the system prompt.

## Setup
To use patterns, first the vector database must be initialized and the necessary files loaded. After that, the LLM can be connected to the model(s) to be queried.

```py
from PBQA import DB, LLM


db = DB(path="examples/db")
db.load_pattern("examples/conversation.yaml")

llm = LLM(db=db, host="192.168.0.137")
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

By default, the DB has a "collection" for each pattern that was loaded. And when unspecified, the default collection from which the history is retrieved when queried, is the pattern name. This can be overridden by specifying the `history_name` parameter in the `llm.ask()` method.

_[Note.](#examples-and-history)_ The use of `n_hist` in conjunction with `n_examples` has not been properly tested yet. Using both parameters may lead to unexpected behavior.

# Function Calling
Another common usecase for LLMSs is function calling. While PBQA doesn't call functions directly, using patterns, it is easy to create valid (json) objects to be used as input for tools. By combining patterns, it is easy to create an agent that can navigate a symbolic systems.

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

llm = LLM(db=db, host="192.168.0.137")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0,
)
```

Creating a class akin to an 'Agent' can be useful for structuring sets of queries or patterns that are designed to accomplish specific tasks.

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

Then, using the `get_forecast` function, the following forecast object is generated:

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

As defined by the [answer_json.yaml](answer_json.yaml) pattern file, the LLM generates a response with a `thought` and `answer` component. The preceding `thought` component allows the LLM to [think](https://arxiv.org/abs/2201.11903) about the provided data before giving a final answer to improve the quality of the response.

Regarding the weather query, note that the properties are based on both the input and the examples provided in the pattern file. While the input did not specify a location, the LLM defaulted to London's latitude and longitude. This is because the examples in the [pattern](weather.yaml) use the coordinates of London whenever no specific location is mentioned.

This concept becomes more powerful when new examples are stored in the database, allowing the LLM to [learn](#feedback) from past interactions and provide more accurate responses in the future.

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


# Feedback
Explanation about n_examples 

## Examples
The `n_examples` parameter is used to specify the number of examples to consider when generating a response. The examples will always consist of `n_examples` entries in the database, ordered by semantic similarity to the input query. Additional keyword arguments can be passed to the `llm.ask()` method to further filter the examples.

### Examples and History
Although using both `n_examples` and `n_hist` in a query should lead to a valid call. The way the entries are formatted currently makes no distinction between examples and history, the history merely being appended after the examples. This may lead to unexpected behavior and poorer response quality when using both parameters.