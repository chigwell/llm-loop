import re

class LLMLoop:
    def __init__(self, model, max_attempts=1000):
        self.model = model
        self.max_attempts = max_attempts

    def query_llm(self, prompt, pattern):
        for attempt in range(self.max_attempts):
            response = self.model(prompt)
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        return None

# Example usage
# Assuming you have a preconfigured LLM model
# llm = AutoModelForCausalLM.from_pretrained(...)
# loop = LLMLoop(llm)
# result = loop.query_llm("Your prompt here", r'Your pattern here')
