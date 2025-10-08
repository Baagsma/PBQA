import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from time import strftime
from typing import Annotated

import requests
from pydantic import BaseModel, Field

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB, LLM  # run with python -m tests.tool_use

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class Weather(BaseModel):
    latitude: float
    longitude: float
    time: Annotated[
        str, Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}$")
    ]


class ThinkAndAnswer(BaseModel):
    thought: str
    answer: str


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


class Agent:
    def __init__(self, db: DB, llm: LLM):
        self.db = db
        self.llm = llm

    def ask_weather(self, input: str):
        weather_query = self.llm.ask(
            input={
                "query": input,
                "now": strftime("%Y-%m-%d %H:%M"),
            },
            pattern="weather",
            model="llama",
            log_input=True,
        )
        weather_obj = weather_query["response"]

        assert (
            type(weather_obj["latitude"]) == float
        ), f"Expected float, got {weather_obj['latitude']} ({type(weather_obj['latitude'])})"

        forecast = get_forecast(**weather_obj)
        forecast = self._format_forecast(forecast)

        print(f"Forecast:\n{json.dumps(forecast, indent=4)}\n")

        return weather_query, self.llm.ask(
            input={
                "query": input,
                "json": json.dumps(forecast),
            },
            pattern="thinkandanswer",
            model="llama",
            log_input=True,
        )

    def _format_forecast(self, forecast: dict):
        return {
            "temperature": f'{forecast["temperature"]} C',
            "precipitation_probability": f'{forecast["precipitation_probability"]} %',
            "precipitation": f'{forecast["precipitation"]} mm',
            "cloud_cover": f'{forecast["cloud_cover"]} % coverage',
        }


db = DB(host="localhost", port=6333, reset=True)
db.load_pattern(
    schema=Weather,
    examples="examples/weather.yaml",
    system_prompt="Your job is to translate the user's input into a weather query object. The object contains the latitude, longitude, and time of the weather query. Reply with the json for the weather query and nothing else.",
    input_key="query",
)
db.load_pattern(
    schema=ThinkAndAnswer,
    examples="examples/answer_info.yaml",
    system_prompt="Use the provided object to answer the query. Leave out any irrelevant information. If the data looks insufficient at face value, try to think step by step to come to a helpful answer. Reply with an answer to the query without mentioning the JSON object or any of its properties. Unless asked, your final answer should not contain any specific numbers or values from the JSON object.",
    input_key="query",
)

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
    temperature=0,
)

agent = Agent(db=db, llm=llm)

weather_query, response = agent.ask_weather("Could I see the stars tonight?")

log.info(f"Weather query:\n{json.dumps(weather_query, indent=4)}")
log.info(f"Response:\n{json.dumps(response, indent=4)}")


response = response["response"]
assert response["thought"] != "", f"Expected a thought, got {response['thought']}"
assert response["answer"] != "", f"Expected an answer, got {response['answer']}"

log.info("All tests passed")
db.delete_collection("weather")
db.delete_collection("thinkandanswer")
