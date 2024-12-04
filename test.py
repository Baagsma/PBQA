import json
from typing import Annotated, List

import requests
from pydantic import BaseModel, Extra, Field
from time import time
from PBQA import DB
import logging

logging.basicConfig(level=logging.INFO)


class Weather(BaseModel):
    latitude: float
    longitude: float
    time: Annotated[
        str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}$")
    ]


schema = Weather.model_json_schema()

print(json.dumps(schema, indent=4))


db = DB(host="localhost", port=6333, reset=True)
db.load_pattern(
    model=Weather,
    examples="examples/temp_weather.yaml",
    system_prompt="Your job is to translate the user's input into a weather query object. The object contains the latitude, longitude, and time of the weather query. Reply with the json for the weather query and nothing else.",
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

# data = {
#     "model": "llama",
#     "id_slot": 0,
#     "cache_prompt": True,
#     "messages": messages,
#     "json_schema": schema,
#     "temperature": 0,
#     "stop": ["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
# }

# try:
#     print(f"Sending request to LLM")
#     then = time()
#     response = requests.post(
#         "http://192.168.0.91:8080/v1/chat/completions",
#         headers={
#             "Content-Type": "application/json",
#             "Authorization": "Bearer no-key",
#         },
#         data=json.dumps(data),
#     )
#     response = response.json()
#     print(f"Response in {time() - then:.3f} s")
#     print(response["choices"][0]["message"]["content"])
#     print(response["usage"])
# except KeyError as e:
#     print(f"Request to LLM failed: {response}")
