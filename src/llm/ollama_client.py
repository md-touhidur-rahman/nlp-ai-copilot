from dataclasses import dataclass
from typing import Optional
import requests


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    temperature: float = 0.2


class OllamaClient:
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    def generate(self, prompt: str) -> str:
        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }

        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["response"].strip()
