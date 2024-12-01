import logging
import os
import sys
from datetime import datetime
from json import dumps
from pathlib import Path
from time import strftime

import requests

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB, LLM  # run with python -m tests.tool_use

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


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
            input=input,
            pattern="weather",
            model="llama",
            external={"now": strftime("%Y-%m-%d %H:%M")},
        )

        assert (
            type(weather_query["latitude"]) == float
        ), f"Expected float, got {weather_query['latitude']} ({type(weather_query['latitude'])})"

        forecast = get_forecast(**weather_query)
        forecast = self._format_forecast(forecast)

        print(f"Forecast:\n{dumps(forecast, indent=4)}\n")

        return self.llm.ask(
            input=input,
            pattern="answer_json",
            model="llama",
            external={"json": dumps(forecast)},
        )

    def _format_forecast(self, forecast: dict):
        return {
            "temperature": f'{forecast["temperature"]}C',
            "precipitation_probability": f'{forecast["precipitation_probability"]}%',
            "precipitation": f'{forecast["precipitation"]}mm',
            "cloud_cover": f'{forecast["cloud_cover"]}% coverage',
        }


db = DB(host="localhost", port=6333, reset=True)
db.load_pattern("examples/weather.yaml")
db.load_pattern("examples/answer_json.yaml")

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0,
)

agent = Agent(db=db, llm=llm)

response = agent.ask_weather("Could I see the stars tonight?")
assert response["thought"] != "", f"Expected a thought, got {response['thought']}"
assert response["answer"] != "", f"Expected an answer, got {response['answer']}"

log.info("All tests passed")
db.delete_collection("weather")
db.delete_collection("answer_json")
