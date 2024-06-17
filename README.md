<h1 align="center">Pattern Based Question and Answer</h1>

## About
Pattern Based Question and Answer (PBQA) is a Python library that provides tools for querying LLMs and managing text embeddings. It combines guided generation with multi-shot prompting to improve response quality and consistency.

 - [Installation](#installation)
 - [Getting Started](#getting-started)
 - [Roadmap](#roadmap)
 - [License](#license-and-acknowledgements)

## Installation
To install the PBQA library, you can clone the repository and install the requirements:

```sh
git clone https://github.com/Baagsma/PBQA.git
pip install -r requirements.txt
```

## Getting Started
First, a database is set up into which your patterns can be loaded. These patterns determine the structure of the LLMs responses and each of its components. The LLM is then connected to the database and the llama.cpp server running the model. When queried, the LLM will then generates a response based on the specified pattern and optional external data.

```python
from PBQA import DB, LLM
from time import strftime

db = DB(path="test/db")
db.load_patterns("test/weather.yaml")

llm = LLM(db=db, host="192.168.0.1")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|end|>"],
    temperature=0,
)

weather_query = llm.ask(
        "Could I see the stars tonight?",
        "weather",
        "llama",
        external={"now": strftime("%Y-%m-%d %H:%M")},
    )
```

Given the [weather.yaml](examples/weather.yaml) pattern file and [llama3](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF) running on 192.168.0.1:8080, `weather_query` should look something like this:

```json
{
    "latitude": 51.51,
    "longitude": 0.13,
    "time": "2024-06-18 01:00",
}
```

For more examples, see the [examples](examples) directory.

## Roadmap
Future features in no particular order with no particular timeline:
    
 - Option to use self-hosted Qdrant server
 - Support for more LLM backends
 - Parallel query execution

## License and Acknowledgements
This project is licensed under the terms of the MIT License. For more details, see the LICENSE file.

Qdrant is a vector database that provides an API for managing and querying text embeddings. The PBQA library uses Qdrant to store and retrieve text embeddings.

llama.cpp is a C++ library that provides an easy-to-use interface for running LLMs on a wide variety of hardware. It includes support for Apple silicon, x86 architectures, and NVIDIA GPUs, as well as custom CUDA kernels for running LLMs on AMD GPUs via HIP. The PBQA library uses llama.cpp to interact with LLMs.

The PBQA library was developed by Bart Haagsma as part of different project. If you have any questions or suggestions, please feel free to contact me at dev.baagsma@gmail.com