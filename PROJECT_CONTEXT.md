# Project Context: MoM Fresh (AI Meeting Intelligence)

## 📌 Project Overview
This project is an AI-powered Meeting Intelligence backend. It ingests meeting audio, transcribes it, identifies speakers (diarization), and uses an LLM to generate structured Minutes of Meeting (MoM) in JSON format. It supports both **static file uploads** and **real-time audio streaming**.

## 🏗️ Architecture & Modules

### 1. Speech Pipeline (`app/speech/pipeline.py`)
- **Core Engine:** Uses `WhisperX` for fast transcription and alignment, and `Pyannote` for speaker diarization.
- **Rules:** We hardcode `language="en"` to force the model to treat Hindi/Hinglish speech as English transliterations (user preference).
- **Output:** Produces a `StructuredTranscript` containing chronological `SpeakerTurn` objects.

### 2. LLM Pipeline (`app/llm/generator.py` & `app/llm/groq_provider.py`)
- **Core Engine:** Uses `Groq` API for ultra-fast LLM inference, strictly outputting JSON matching the `MeetingMinutes` schema.
- **Provider Abstraction:** The system is designed so `GroqProvider` can be easily swapped with a local LLM in the future without breaking the `MoMGenerator`.

### 3. FastAPI Backend (`app/api/router.py` & `app/api/server.py`)
- **REST API:** `POST /meetings/process` accepts `.wav`/`.mp3` files, runs the full pipeline, and returns the transcript + MoM.
- **WebSocket API:** `ws://localhost:8000/meetings/stream` handles real-time live meetings.

### 4. Real-Time Streaming Session (`app/meeting/streaming.py`)
- **Buffer & Chunks:** Buffers binary audio from the WebSocket. Processes audio in strict **60-second chunks** to ensure chronological consistency.
- **Overlap Handling:** Keeps a 5-second overlap between chunks so words at the boundary aren't cut off. The overlap is removed from the transcript delta so the LLM doesn't see duplicates.
- **Thread Safety:** Uses `threading.Lock()` for the audio buffer and `asyncio.to_thread()` in the router so the heavy PyTorch inference never blocks the WebSocket from receiving new audio packets.
- **Incremental MoM:** Uses a special `INCREMENTAL_SYSTEM_PROMPT` to pass the *previous* MoM JSON + the *new* transcript delta, instructing the LLM to cleanly merge new points without duplicating old ones.

### 5. Live Client (`live_microphone_client.py`)
- A standalone Python script using `pyaudio` and `websocket-client` that captures desktop microphone audio and streams it to the backend to test the live real-time MoM generation.

## ⚙️ Environment & Setup
- **Virtual Environment:** Sourced at `whisperX/.venv`
- **Config:** `.env` file controls all settings.
- **Telemetry Bug:** `opentelemetry-api` is pinned to `1.39.1` to prevent a dependency crash between `fastapi` and `pyannote`.

## 🚀 Future Roadmap: NVIDIA DGX Spark GPU
The current pipeline runs on CPU, which takes ~75 seconds to process 60 seconds of audio. To migrate to the DGX GPU for real-time (< 5s) latency:
1. Update `.env`: set `DEVICE=cuda` and `COMPUTE_TYPE=float16`.
2. Ensure CUDA-enabled PyTorch is installed in the `.venv` (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).
