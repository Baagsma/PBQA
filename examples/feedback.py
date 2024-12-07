from json import dumps
from time import strftime
from typing import Annotated

from pydantic import BaseModel, Field

from PBQA import DB, LLM


class Weather(BaseModel):
    latitude: float
    longitude: float
    time: Annotated[
        str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}$")
    ]


db = DB("examples/db")
db.load_pattern(
    schema=Weather,
    examples="examples/weather.yaml",
    system_prompt="Your job is to translate the user's input into a weather query object. The object contains the latitude, longitude, and time of the weather query. Reply with the json for the weather query and nothing else.",
    input_key="query",
)

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
    temperature=0,
)

initial_response = llm.ask(
    input={
        "query": "Is it going to rain tonight at home?",
        "now": strftime("%Y-%m-%d %H:%M"),
    },
    pattern="weather",
    model="llama",
    n_example=2,
    **{"metadata.feedback": True},
)["response"]

print(f"Initial response:\n{dumps(initial_response, indent=4)}\n")

db.add(
    input="What's the weather like at home?",
    collection_name="weather",
    latitude=48.86,
    longitude=2.35,
    now="2021-03-02 14:46",
    time="2021-03-02 14:46",
    feedback=True,
)

# See the repetition caveat in the README
# db.add(
#     input="Is it going to rain tonight at home?",
#     collection_name="weather",
#     latitude=48.86,
#     longitude=2.35,
#     now="2021-03-02 14:46",
#     time="2021-03-02 21:00",
#     feedback=True,
# )

feedback_response = llm.ask(
    input={
        "query": "Is it going to rain tonight at home?",
        "now": strftime("%Y-%m-%d %H:%M"),
    },
    pattern="weather",
    model="llama",
    n_example=2,
    **{"metadata.feedback": True},
)["response"]

print(f"Feedback response:\n{dumps(feedback_response, indent=4)}\n")
