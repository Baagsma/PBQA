from PBQA import DB, LLM
from time import strftime
import datetime
from json import dumps

db = DB(path="examples/db")
db.load_pattern("examples/weather.yaml")

llm = LLM(db=db, host="192.168.0.137")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0,
)

initial_response = llm.ask(
    input="Is it going to rain tonight at home?",
    pattern="weather",
    model="llama",
    external={"now": strftime("%Y-%m-%d %H:%M")},
    n_example=1,
    feedback=True,
)

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
    input="Is it going to rain tonight at home?",
    pattern="weather",
    model="llama",
    external={"now": strftime("%Y-%m-%d %H:%M")},
    n_example=1,
    feedback=True,
)

print(f"Feedback response:\n{dumps(feedback_response, indent=4)}\n")
