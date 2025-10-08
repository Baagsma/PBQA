import logging

from pydantic import BaseModel

from PBQA import DB, LLM


class Conversation(BaseModel):
    reply: str


db = DB("examples/db")
db.load_pattern(
    schema=Conversation,
    system_prompt="You are a virtual assistant. You are here to help where you can or simply engage in conversation.",
)

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
)

while True:
    user_input = input("User\n> ")

    response = llm.ask(
        input=user_input,
        pattern="conversation",
        model="llama",
        n_hist=50,
    )["response"]

    db.add(
        input=user_input,
        collection_name="conversation",
        **response,
    )

    print(f"\nAssistant\n> {response['reply']}\n")
