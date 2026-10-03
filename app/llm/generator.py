import logging
from app.llm.schema import MeetingMinutes
from app.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI Meeting Assistant. Your task is to extract structured Minutes of Meeting (MoM) from a chronological speaker-aware transcript.

INSTRUCTIONS:
1. The transcript may contain English, Hindi, or Hinglish code-switching. Understand all languages but ALWAYS generate the final MoM entirely in English.
2. Identify participants exclusively using the provided speaker IDs (e.g., SPEAKER_00, SPEAKER_01).
3. Distinguish between mere suggestions and confirmed decisions.
4. Extract Action Items. If an owner or deadline is not explicitly supported by the transcript, return null. Do NOT invent or guess information.
5. Provide the output strictly as a JSON object matching the required schema.

REQUIRED JSON SCHEMA:
{
  "meeting_title": "String",
  "summary": "String",
  "participants": ["String"],
  "key_discussion_points": ["String"],
  "decisions": ["String"],
  "action_items": [
    {
      "task": "String",
      "owner": "String or null",
      "deadline": "String or null"
    }
  ],
  "important_dates": ["String"],
  "open_questions": ["String"]
}
"""

class MoMGenerator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate(self, transcript_text: str) -> MeetingMinutes:
        if not transcript_text or not transcript_text.strip():
            raise ValueError("Transcript is empty.")
            
        logger.info("Starting MoM generation...")
        
        # We can implement chunking here later if transcript is too long.
        # For now, standard sized meetings are sent directly.
        user_prompt = f"Here is the chronological speaker-aware transcript:\n\n{transcript_text}\n\nGenerate the structured Minutes of Meeting in JSON."
        
        raw_json_dict = self.provider.generate_mom(SYSTEM_PROMPT, user_prompt)
        
        # Validate schema
        logger.info("Validating MoM JSON schema using Pydantic...")
        mom = MeetingMinutes(**raw_json_dict)
        return mom
