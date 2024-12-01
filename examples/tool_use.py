from datetime import datetime
from json import dumps
from time import strftime

import requests

from PBQA import DB, LLM


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

        print(f"Query:\n{dumps(weather_query, indent=4)}\n")

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


db = DB(path="examples/db")
db.load_pattern("examples/weather.yaml")
db.load_pattern("examples/answer_json.yaml")

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
    temperature=0,
)

agent = Agent(db=db, llm=llm)

response = agent.ask_weather("Could I see the stars tonight?")
print(f"Response:\n{dumps(response, indent=4)}")
