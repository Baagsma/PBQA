<h1 align="center">Pattern Based Question and Answer</h1>

## Description
Pattern Based Question and Answer (PBQA) is a Python library that provides tools for querying LLMs and managing text embeddings. It combines [guided generation](examples/README.md#grammar) with [multi-shot prompting](https://arxiv.org/abs/2005.14165) to improve response quality and ensure consistency. By enforcing valid responses, PBQA makes it easy to combine the flexibility of LLMs with the reliability and control of symbolic approaches. 

 - [Installation](#installation)
 - [Usage](#usage)
 - [Roadmap](#roadmap)
 - [Relevant Literature](#relevant-literature)
 - [Contributing](#contributing)
  - [Support](#support)
 - [License](#license-and-acknowledgements)

## Installation
PBQA requires Python 3.9 or higher, and can be installed via pip:

```sh
pip install PBQA
```

Additionally, PBQA requires a running instance of llama.cpp to interact with LLMs. For instructions on installation, see the [llama.cpp repository](https://github.com/ggerganov/llama.cpp/tree/master?tab=readme-ov-file#usage).

## Usage
### llama.cpp
For instructions on hosting a model with llama.cpp, see the [following page](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#quick-start). Optionally, [caching](#cache) can be enabled speed up generation.

### Python
PBQA provides a simple API for querying LLMs.

```python
from PBQA import DB, LLM
from time import strftime

# First, we set up a database at a specified path
db = DB(path="examples/db")
# Then, we load a pattern file into the database
db.load_pattern("examples/weather.yaml")

# Next, we connect to the LLM server
llm = LLM(db=db, host="192.168.0.1")
# And connect to the model
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>"],
    temperature=0,
)

# Finally, we query the LLM and receive a response based on the specified pattern
# Optionally, external data can be provided to the LLM which it can use in its response
weather_query = llm.ask(
        "Could I see the stars tonight?",
        "weather",
        "llama",
        external={"now": strftime("%Y-%m-%d %H:%M")},
    )
```

Using the [weather.yaml](examples/weather.yaml) pattern file and [llama3](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF) running on 192.168.0.1:8080, the response should look something like this:

```json
{
    "latitude": 51.51,
    "longitude": 0.13,
    "time": "2024-06-18 01:00",
}
```

For more information, see the [examples](examples/README.md) directory.

### Pattern Files
Pattern files are used to guide the LLM in generating responses. They are written in YAML and consist of three parts: the system prompt, component metadata, and examples.

```yaml
# The system prompt is the main instruction given to the LLM telling it what to do
system_prompt: Your job is to translate the user's input into a weather query. Reply with the json for the weather query and nothing else.
now:  # Each component of the response needs to have it's own key, "component:" at minimum
  external: true  # Optionally, specify whether the component requires external data
latitude:
  grammar: |  # Or define a GBNF grammar
    root         ::= coordinate
    coordinate   ::= integer "." integer
    integer      ::= digit | digit digit
    digit        ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
longitude:
  grammar: ...
time:
  grammar: ...
examples:  # Lastly, examples can be provided for multi-shot prompting
- input: What will the weather be like tonight
  now: 2019-09-30 10:36
  latitude: 51.51
  longitude: 0.13
  time: 2019-09-30 20:00
- input: Could I see the stars tonight?
  ...
```

For more examples, look at the pattern files in the [examples](examples/README.md#patterns) directory. Information on the GBNF grammar format can be found [here](https://github.com/ggerganov/llama.cpp/tree/master/grammars#gbnf-guide).

### Cache
Unless overridden, queries using the same pattern will use the same system prompt and base examples This allows a large part of the response to be cached, speeding up generation. This can be disabled by setting `use_cache=False` in the `ask()` method.

_Note:_ To cache the patterns, PBQA will try allocate a slot/process for each pattern-model pair in the llama.cpp server. As such, make sure to set `-np` to the amount of unique combinations of patterns and models you want to enable caching for.

## Roadmap
Future features in no particular order with no particular timeline:

 - Preset grammars for common data types
 - Option to pick a specific caching slot for a query
 - Option to use self-hosted Qdrant server
 - Parallel query execution
 - Combining multi-shot prompting with message history
 - Multimodal support
 - Further speed improvements (possibly [batching](https://github.com/guidance-ai/guidance?tab=readme-ov-file#guidance-acceleration))
 - Support for more LLM backends

## Relevant Literature

 - [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
 - [Many-Shot In-Context Learning (Aragwal, 2024)](https://arxiv.org/abs/2404.11018)
 - [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)

## Contributing
Contributions are welcome! If you have any suggestions or would like to contribute, please open an issue or a pull request.

## Support
If you want to support the development of PBQA, consider [buying me a coffee](https://ko-fi.com/baagsma). Any support is greatly appreciated!

## License and Acknowledgements
This project is licensed under the terms of the MIT License. For more details, see the LICENSE file.

Qdrant is a vector database that provides an API for managing and querying text embeddings. PBQA uses Qdrant to store and retrieve text embeddings.

llama.cpp is a C++ library that provides an easy-to-use interface for running LLMs on a wide variety of hardware. It includes support for Apple silicon, x86 architectures, and NVIDIA GPUs, as well as custom CUDA kernels for running LLMs on AMD GPUs via HIP. PBQA uses llama.cpp to interact with LLMs.

PBQA was developed by Bart Haagsma as part of different project. If you have any questions or suggestions, please feel free to contact me at dev.baagsma@gmail.com