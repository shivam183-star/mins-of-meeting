import os
import tempfile
import logging
import json
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Any, Dict

from app.speech.pipeline import SpeechPipeline, format_llm_transcript
from app.llm.generator import MoMGenerator
from app.llm.schema import MeetingMinutes
from app.llm.groq_provider import GroqProvider
from app.meeting.streaming import StreamingMeetingSession

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
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temp file {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

@router.websocket("/meetings/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    session = StreamingMeetingSession(speech_pipeline, llm_provider)
    logger.info("WebSocket connection established for streaming meeting.")
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                # Binary audio data
                session.add_audio(message["bytes"])
                
                # Extract 60s chunk if ready without blocking
                chunk_data = session.extract_chunk_if_ready(force=False)
                if chunk_data:
                    async def process_and_send(data):
                        try:
                            result = await asyncio.to_thread(session._process_chunk, data, False)
                            if result:
                                await websocket.send_json({"type": "update", "data": result})
                        except Exception as e:
                            logger.error(f"Background processing error: {e}")
                    
                    # Fire and forget the processing task
                    asyncio.create_task(process_and_send(chunk_data))
                    
            elif "text" in message and message["text"]:
                event = json.loads(message["text"])
                if event.get("type") == "end_meeting":
                    logger.info("Received end_meeting event. Finalizing...")
                    
                    # Process remaining backlog in exactly 60s chunks
                    while True:
                        # Wait if a chunk is currently being processed
                        while session.is_processing:
                            await asyncio.sleep(1)
                            
                        chunk_data = session.extract_chunk_if_ready(force=False)
                        if chunk_data:
                            result = await asyncio.to_thread(session._process_chunk, chunk_data, False)
                            await websocket.send_json({"type": "update", "data": result})
                        else:
                            break
                    
                    # Finally process whatever is left (< 60s)
                    chunk_data = session.extract_chunk_if_ready(force=True)
                    if chunk_data:
                        result = await asyncio.to_thread(session._process_chunk, chunk_data, True)
                        await websocket.send_json({"type": "final_update", "data": result})
                    else:
                        await websocket.send_json({"type": "final_update", "data": None})
                    
                    await websocket.close()
                    break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011)
        except:
            pass
