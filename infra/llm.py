from openai import OpenAI
from typing import List, Dict, Optional
from infra.config import Config

class LLM:
    def __init__(self, config: Config):
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        self.model = config.llm_model
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """调用LLM生成回复"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """简化版生成接口"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)
