from configs.params import ModelParams
import openai
model_config = ModelParams()
class LLMGenerate():
    def __init__(self):
        self.llm = openai.AzureOpenAI(
            api_key=model_config.chat_key,
            azure_endpoint=model_config.chat_endpoint,
            api_version="2024-02-01",
            azure_deployment="prj-taiho-gpt4o",
        )
    def generate(self, content):
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": content}],
        )
        return response.choices[0].message.content
