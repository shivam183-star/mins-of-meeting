from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class Word:
    word: str
    start: float
    end: float
    score: Optional[float] = None
    speaker: Optional[str] = None

@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float
    text: str
    words: List[Word]

@dataclass
class MeetingMetadata:
    duration: float
    language: str

@dataclass
class Speaker:
    speaker_id: str

@dataclass
class StructuredTranscript:
    meeting_metadata: MeetingMetadata
    speakers: List[Speaker]
    turns: List[SpeakerTurn]

    def to_dict(self):
        return asdict(self)
