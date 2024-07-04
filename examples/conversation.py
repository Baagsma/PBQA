from PBQA import DB, LLM
import logging

logging.basicConfig(level=logging.INFO)


db = DB(path="examples/db")
db.load_pattern("examples/conversation.yaml")

llm = LLM(db=db, host="192.168.0.137")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
)

while True:
    user_input = input("User\n> ")

    hist = db.where(collection_name="conversation", n=10)
    hist.reverse()

    hist = [{"input": item["input"], "reply": item["reply"]} for item in hist]

    response = llm.ask(
        input=user_input,
        pattern="conversation",
        model="llama",
        external={"hist": hist},
        n_hist=50,
        return_external=True,
    )

    db.add(
        input=user_input,
        collection_name="conversation",
        **response,
    )

    print(f"\nAssistant\n> {response['reply']}\n")
