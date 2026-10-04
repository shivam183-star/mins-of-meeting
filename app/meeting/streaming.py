import os
import time
import json
import logging
import tempfile
import wave
import threading
from typing import Optional

from app.speech.pipeline import SpeechPipeline, format_llm_transcript
from app.speech.schema import StructuredTranscript, SpeakerTurn
from app.llm.provider import LLMProvider
from app.llm.schema import MeetingMinutes

logger = logging.getLogger(__name__)

INCREMENTAL_SYSTEM_PROMPT = """You are an expert AI Meeting Assistant. Your task is to update an existing structured Minutes of Meeting (MoM) with new chronological speaker-aware transcript data.

INSTRUCTIONS:
1. Merge the new information into the existing MoM structure.
2. DO NOT duplicate existing points. If a new point is already covered, ignore it.
3. If a new decision, action item, or discussion point arises, add it to the respective list.
4. Update the summary if necessary to reflect the broader context.
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

class StreamingMeetingSession:
    def __init__(self, speech_pipeline: SpeechPipeline, llm_provider: LLMProvider):
        self.speech_pipeline = speech_pipeline
        self.llm_provider = llm_provider
        
        self.audio_buffer = bytearray()
        self.sample_width = 2
        self.channels = 1
        self.framerate = 16000
        
        self.chunk_duration_sec = 60
        self.overlap_sec = 5
        
        self.processed_duration = 0.0
        self.chunk_index = 0
        
        self.current_mom: Optional[MeetingMinutes] = None
        self.global_speaker_mapping = {}
        self.global_speaker_counter = 0
        self.buffer_lock = threading.Lock()
        self.is_processing = False

    def add_audio(self, pcm_data: bytes):
        with self.buffer_lock:
            self.audio_buffer.extend(pcm_data)

    def extract_chunk_if_ready(self, force=False) -> Optional[bytes]:
        with self.buffer_lock:
            bytes_per_sec = self.framerate * self.channels * self.sample_width
            current_buffer_sec = len(self.audio_buffer) / bytes_per_sec
            
            if current_buffer_sec >= self.chunk_duration_sec or (force and current_buffer_sec > 0):
                if self.is_processing:
                    return None # Still processing previous chunk, let buffer grow
                
                if force:
                    chunk_bytes_len = len(self.audio_buffer)
                else:
                    chunk_bytes_len = int(self.chunk_duration_sec * bytes_per_sec)
                
                chunk_data = self.audio_buffer[:chunk_bytes_len]
                
                if force:
                    self.audio_buffer = bytearray()
                else:
                    overlap_bytes = self.overlap_sec * bytes_per_sec
                    # Keep the last `overlap_sec` of the chunk we just extracted, PLUS the remaining unextracted backlog
                    self.audio_buffer = self.audio_buffer[chunk_bytes_len - overlap_bytes:]
                        
                self.is_processing = True
                return chunk_data
        return None

    def process_if_ready(self, force=False) -> Optional[dict]:
        chunk_data = self.extract_chunk_if_ready(force)
        if chunk_data is None:
            return None
        return self._process_chunk(chunk_data, is_final=force)

    def _process_chunk(self, chunk_data: bytes, is_final=False) -> dict:
        try:
            start_time = time.time()
            self.chunk_index += 1
            
            bytes_per_sec = self.framerate * self.channels * self.sample_width
            chunk_bytes_len = len(chunk_data)

            # Write to temp file
            fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, 'wb') as f:
                with wave.open(f, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.sample_width)
                    wf.setframerate(self.framerate)
                    wf.writeframes(chunk_data)
                    
            # 1. Speech Pipeline
            speech_start = time.time()
            transcript = self.speech_pipeline.process_audio(temp_file_path)
            os.remove(temp_file_path)
            speech_latency = time.time() - speech_start
            
            # 2. Speaker Consistency & Overlap Truncation
            delta_transcript = self._resolve_delta_transcript(transcript, is_final)
            
            audio_duration = chunk_bytes_len / bytes_per_sec
            
            # 3. LLM Incremental Update
            llm_start = time.time()
            llm_text = format_llm_transcript(delta_transcript)
            
            if llm_text.strip():
                if not self.current_mom:
                    # Generate first MoM
                    from app.llm.generator import SYSTEM_PROMPT
                    user_prompt = f"Here is the first part of the chronological speaker-aware transcript:\n\n{llm_text}\n\nGenerate the structured Minutes of Meeting in JSON."
                    raw_json = self.llm_provider.generate_mom(SYSTEM_PROMPT, user_prompt)
                    self.current_mom = MeetingMinutes(**raw_json)
                else:
                    # Update existing MoM
                    user_prompt = f"Existing MoM:\n{self.current_mom.model_dump_json(indent=2)}\n\nNew Transcript Delta:\n{llm_text}\n\nReturn the updated structured Minutes of Meeting in JSON."
                    raw_json = self.llm_provider.generate_mom(INCREMENTAL_SYSTEM_PROMPT, user_prompt)
                    self.current_mom = MeetingMinutes(**raw_json)
                
            llm_latency = time.time() - llm_start
            processing_time = time.time() - start_time
            
            logger.info(f"Chunk {self.chunk_index}: Audio {audio_duration:.2f}s | Speech {speech_latency:.2f}s | LLM {llm_latency:.2f}s | Total {processing_time:.2f}s")
            
            # Update processed duration for next chunk timestamps
            if not is_final:
                self.processed_duration += (audio_duration - self.overlap_sec)
            else:
                self.processed_duration += audio_duration

            return {
                "chunk_index": self.chunk_index,
                "audio_duration": audio_duration,
                "speech_latency": speech_latency,
                "llm_latency": llm_latency,
                "total_latency": processing_time,
                "delta_transcript": delta_transcript.to_dict(),
                "mom": self.current_mom.model_dump() if self.current_mom else None
            }
        finally:
            with self.buffer_lock:
                self.is_processing = False
        
    def _resolve_delta_transcript(self, transcript: StructuredTranscript, is_final: bool) -> StructuredTranscript:
        # Filter out turns that are entirely within the overlap period of the *previous* chunk
        # Since we passed the overlap audio again, the first 5 seconds of this chunk 
        # correspond to the last 5 seconds of the previous chunk. We should drop them so LLM doesn't see duplicates.
        filtered_turns = []
        
        if self.chunk_index > 1:
            overlap_threshold = self.overlap_sec - 0.5 # Add a small margin
        else:
            overlap_threshold = 0.0
            
        for turn in transcript.turns:
            if turn.end <= overlap_threshold:
                continue # Skip this turn, it was covered in the previous chunk's tail
                
            # Naive mapping: assume SPEAKER_00 is always the same person.
            if turn.speaker not in self.global_speaker_mapping:
                self.global_speaker_mapping[turn.speaker] = f"SPEAKER_{self.global_speaker_counter:02d}"
                self.global_speaker_counter += 1
            turn.speaker = self.global_speaker_mapping[turn.speaker]
            
            # Shift timestamps
            # Since we dropped the overlapping audio conceptually from the timestamp,
            # this chunk's time 0 is actually self.processed_duration - self.overlap_sec (if chunk > 1)
            # Actually, the simplest is:
            time_offset = self.processed_duration
            if self.chunk_index > 1:
                time_offset -= self.overlap_sec
                
            turn.start += time_offset
            turn.end += time_offset
            
            filtered_turns.append(turn)
            
        transcript.turns = filtered_turns
        return transcript
