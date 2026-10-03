import os
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from app.speech.pipeline import SpeechPipeline, format_llm_transcript
from app.llm.generator import MoMGenerator
from app.llm.schema import MeetingMinutes
from app.llm.groq_provider import GroqProvider

logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate the pipeline components globally so models are cached in RAM
speech_pipeline = SpeechPipeline()
llm_provider = GroqProvider()
mom_generator = MoMGenerator(llm_provider)

class ProcessResponse(BaseModel):
    transcript_json: Dict[str, Any]
    mom: MeetingMinutes

from fastapi.responses import RedirectResponse

@router.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.post("/meetings/process", response_model=ProcessResponse)
async def process_meeting(audio_file: UploadFile = File(...)):
    if not audio_file.filename.endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Only .wav, .mp3, .m4a files are supported.")
        
    temp_file_path = None
    try:
        # 1. Save uploaded file temporarily
        fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, 'wb') as f:
            f.write(await audio_file.read())
            
        logger.info(f"Processing uploaded file {audio_file.filename} at {temp_file_path}")
        
        # 2. Run Speech Pipeline
        logger.info("Running speech pipeline...")
        transcript = speech_pipeline.process_audio(temp_file_path)
        
        # 3. Format Transcript for LLM
        llm_text = format_llm_transcript(transcript)
        
        # 4. Generate MoM
        logger.info("Generating Minutes of Meeting...")
        mom_result = mom_generator.generate(llm_text)
        
        # 5. Return JSON payload
        return ProcessResponse(
            transcript_json=transcript.to_dict(),
            mom=mom_result
        )
        
    except Exception as e:
        logger.error(f"Error processing meeting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process meeting: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temp file {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")
