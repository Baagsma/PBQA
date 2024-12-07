import json
from typing import Annotated, List

import requests
from pydantic import BaseModel, Extra, Field
from time import time, strftime
from PBQA import DB, LLM
import logging

logging.basicConfig(level=logging.INFO)


class Weather(BaseModel):
    latitude: float
    longitude: float
    time: Annotated[
        str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}$")
    ]


schema = Weather.model_json_schema()

db = DB(host="localhost", port=6333, auto_update=True)
db.load_pattern(
    schema=Weather,
    examples="examples/temp_weather.yaml",
    system_prompt="Your job is to translate the user's input into a weather query object. The object contains the latitude, longitude, and time of the weather query. Reply with the json for the weather query and nothing else.",
    input_key="query",
)

most_similar = db.query(
    "weather",
    "What's the weather like in London?",
    n=1,
)
print(json.dumps(most_similar, indent=4))

most_recent = db.where(
    "weather",
    n=3,
)
print(json.dumps(most_recent, indent=4))


llm = LLM(db=db, host="192.168.0.91")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
)

llm.link("weather", "llama")

weather_query = llm.ask(
    input={
        "query": "What will the weather be like tomorrow at 10:00?",
        "now": strftime("%Y-%m-%d %H:%M"),
    },
    pattern="weather",
)

print(json.dumps(weather_query, indent=4))
