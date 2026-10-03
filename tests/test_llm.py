import pytest
import os
from unittest.mock import Mock, patch
from pydantic import ValidationError
from app.llm.schema import MeetingMinutes
from app.llm.generator import MoMGenerator
from app.llm.provider import LLMProvider

class MockProvider(LLMProvider):
    def generate_mom(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "meeting_title": "Test Meeting",
            "summary": "This is a mock summary.",
            "participants": ["SPEAKER_00"],
            "key_discussion_points": ["Point 1"],
            "decisions": ["Decision 1"],
            "action_items": [
                {"task": "Task 1", "owner": "SPEAKER_00", "deadline": "Tomorrow"}
            ],
            "important_dates": [],
            "open_questions": []
        }

def test_mom_generator_valid():
    provider = MockProvider()
    generator = MoMGenerator(provider)
    
    mom = generator.generate("[00:00 - 00:05] SPEAKER_00:\nHello")
    assert isinstance(mom, MeetingMinutes)
    assert mom.meeting_title == "Test Meeting"
    assert len(mom.action_items) == 1
    assert mom.action_items[0].owner == "SPEAKER_00"

def test_mom_generator_empty_transcript():
    provider = MockProvider()
    generator = MoMGenerator(provider)
    
    with pytest.raises(ValueError, match="Transcript is empty"):
        generator.generate("   \n  ")

def test_groq_provider_missing_key(monkeypatch):
    monkeypatch.setattr("app.config.config.GROQ_API_KEY", None)
    
    from app.llm.groq_provider import GroqProvider
    with pytest.raises(ValueError, match="GROQ_API_KEY environment variable is not set"):
        GroqProvider()
