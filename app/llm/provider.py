from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers (e.g., Groq, Local).
    """
    @abstractmethod
    def generate_mom(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generate structured MoM JSON using the provider.
        Should return a dictionary containing the JSON response.
        """
        pass
