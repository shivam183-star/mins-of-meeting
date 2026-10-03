import json
import logging
from groq import Groq
from app.llm.provider import LLMProvider
from app.config import config

logger = logging.getLogger(__name__)

class GroqProvider(LLMProvider):
    def __init__(self):
        self.api_key = config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = config.GROQ_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS

    def generate_mom(self, system_prompt: str, user_prompt: str) -> dict:
        logger.info(f"Generating MoM using Groq API with model {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
