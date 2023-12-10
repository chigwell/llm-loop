import os
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from llm_loop.main import LLMLoop

# Model defaults based on your provided example
model_name = "TheBloke/CodeLlama-13B-Instruct-GGUF"
model_file = "codellama-13b-instruct.Q2_K.gguf"

current_user = os.environ.get('USER', 'default_user')
start_dir = f'/Users/{current_user}/.cache/gpt4all/TheBloke/CodeLlama-13B-Instruct-GGUF'

# Initialize the model
model_path = f"{start_dir}/{model_file}"

llm = AutoModelForCausalLM.from_pretrained(model_name, model_file=model_path, model_type='mistral', gpu_layers=1)

# Initialize LLMLoop with the model
loop = LLMLoop(llm, 10)

# Define your prompt
prompt = "Write a possibility that Trump will be a president again. Write only a number from 0 to 1." \
         "Please provide your answer in the following format: `probability: <your_number_here>`. For example, `probability: 0.75`."

# Define a regex pattern to match the response
pattern = r'probability: ([0-9.]+)'

response = loop.query_llm(prompt=prompt, pattern=pattern)

print("Response:", response)
