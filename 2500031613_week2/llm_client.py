
from abc import ABC, abstractmethod
import requests

try:
    from google import genai
except ImportError:
    genai = None


# Common LLM interface
class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


# Ollama local provider
class OllamaProvider(LLMProvider):

    def __init__(self, model="llama3.2",
                 url="http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=180,
        )

        response.raise_for_status()
        return response.json()["response"]


# Gemini cloud provider
class GeminiProvider(LLMProvider):

    def __init__(self, api_key: str, model="gemini-3.5-flash-lite"):

        if genai is None:
            raise RuntimeError(
                "Install google-genai to use Gemini."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text or ""


# Common LLM client
class LLMClient:

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, prompt: str) -> str:
        return self.provider.generate(prompt)
