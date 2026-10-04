import json
import logging
from openai import OpenAI
from app.llm.provider import LLMProvider
from app.config import config

logger = logging.getLogger(__name__)

class LocalProvider(LLMProvider):
    def __init__(self):
        self.base_url = config.LOCAL_LLM_BASE_URL
        self.api_key = config.LOCAL_LLM_API_KEY
        
        logger.info(f"Initializing Local LLM Provider at {self.base_url}")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.model = config.LOCAL_LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS

    def generate_mom(self, system_prompt: str, user_prompt: str) -> dict:
        logger.info(f"Generating MoM using Local LLM with model {self.model}...")
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
            logger.error(f"Local LLM API error: {e}")
            raise
