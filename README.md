# llm-loop

## Overview

`llm-loop` is a Python package designed to simplify the process of querying language models (like GPT or similar models) until a response matching a specified pattern is obtained or a maximum number of attempts is reached. This is particularly useful when working with AI models in scenarios where a specific format of response is required.

## Installation

To install `llm-loop`, you will need to clone the repository and install the dependencies. Ensure you have Python 3.x installed on your system before proceeding.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chigwell/llm-loop.git
   cd llm-loop
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install the necessary Python packages, including `ctransformers` and any other dependencies.

## Usage

Here's a basic example of how to use `llm-loop`:

1. **Import the necessary modules:**
   ```python
   import os
   from ctransformers import AutoModelForCausalLM, AutoTokenizer
   from llm_loop.main import LLMLoop
   ```

2. **Initialize the model with custom parameters:**
   ```python
   model_name = "YourModelName"
   model_file = "YourModelFileName"
   start_dir = '/path/to/your/model'
   model_path = f"{start_dir}/{model_file}"

   llm = AutoModelForCausalLM.from_pretrained(model_name, model_file=model_path, model_type='YourModelType', gpu_layers=YourGPULayers)
   ```

3. **Create an instance of LLMLoop and query the model:**
   ```python
   loop = LLMLoop(llm, 10)  # 10 is the maximum number of attempts

   prompt = "Your prompt here"
   pattern = r'Your regex pattern here'

   response = loop.query_llm(prompt=prompt, pattern=pattern)

   print("Response:", response)
   ```

## Contributing

Contributions to `llm-loop` are welcome! Please feel free to submit pull requests or open issues to suggest improvements or add new features.

## License

MIT.
