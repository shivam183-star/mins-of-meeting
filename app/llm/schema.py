from pydantic import BaseModel, Field
from typing import List, Optional

class ActionItem(BaseModel):
    task: str = Field(description="The action item task.")
    owner: Optional[str] = Field(description="The speaker assigned to the task (e.g., SPEAKER_00). Null if not explicitly assigned.", default=None)
    deadline: Optional[str] = Field(description="Deadline for the task. Null if not specified.", default=None)

class MeetingMinutes(BaseModel):
    meeting_title: str = Field(description="A concise title for the meeting.")
    summary: str = Field(description="A brief summary of what was discussed.")
    participants: List[str] = Field(description="List of speakers identified in the meeting (e.g., SPEAKER_00).")
    key_discussion_points: List[str] = Field(description="Key points discussed during the meeting.")
    decisions: List[str] = Field(description="Decisions that were explicitly made. Do not include suggestions.")
    action_items: List[ActionItem] = Field(description="Action items assigned during the meeting.")
    important_dates: List[str] = Field(description="Any important dates mentioned.")
    open_questions: List[str] = Field(description="Questions raised but not resolved.")
