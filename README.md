# MoM Fresh (AI Meeting Intelligence)

An AI-powered Meeting Intelligence backend that ingests meeting audio, transcribes it using WhisperX, identifies speakers via Pyannote (diarization), and generates structured Minutes of Meeting (MoM) in JSON format using an LLM (Groq API).

Supports both **static file uploads** (for pre-recorded meetings) and **real-time audio streaming**.

---

## 🚀 Features
- **Accurate Transcription & Alignment:** Powered by WhisperX.
- **Speaker Diarization:** Identifies speakers using Pyannote Audio.
- **Hinglish/Code-switching Support:** Forces English transcription to gracefully handle Hindi/Hinglish speech.
- **Ultra-fast LLM MoM Generation:** Uses the Groq API to generate structured JSON outputs.
- **Real-time Live Streaming:** Processes audio in 60-second chunks over WebSocket for live meetings.
- **Scalable Architecture:** Easily swap out the LLM provider (e.g., to a local model) when running on high-end hardware.

---

## 💻 Installation & Usage: CPU PC (Local Testing)
Use these instructions if you are testing on a standard laptop/PC without a dedicated high-end GPU. Note that processing will be slower.

### 1. Prerequisites
- Python 3.10+
- FFmpeg installed and added to your system PATH.
- Hugging Face Account (for Pyannote models)
- Groq Account (for LLM API)

### 2. Setup Environment
Clone the repository and set up your virtual environment:

```bash
git clone https://github.com/your-repo/mom-fresh.git
cd mom-fresh
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration
Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```
Edit `.env` to include your specific tokens:
- `HF_TOKEN`: Get this from Hugging Face. You must also accept the user conditions for `pyannote/speaker-diarization-community-1` and `pyannote/segmentation-3.0` on the Hugging Face website.
- `GROQ_API_KEY`: Get this from the Groq console.
- Leave `DEVICE=cpu` and `COMPUTE_TYPE=int8` (or `float32`) for CPU execution.

### 4. Running the Server
Start the FastAPI backend:
```bash
python start_server.py
```
The server will be available at `http://localhost:8000`.

### 5. Testing
- **Static File:** Send a POST request to `http://localhost:8000/meetings/process` with a `.wav` or `.mp3` file.
- **Live Microphone:** Run the standalone client in a separate terminal:
  ```bash
  python live_microphone_client.py
  ```

---

## ⚡ Installation & Usage: NVIDIA DGX Spark (High Performance)
Use these instructions when deploying on an NVIDIA DGX or any high-end server with a CUDA-enabled GPU for real-time inference and processing of long meeting files.

### 1. Prerequisites
- NVIDIA GPU with CUDA installed.
- Python 3.10+
- FFmpeg installed.

### 2. Setup Environment
Clone the repository and install PyTorch with CUDA support **before** installing the rest of the requirements:

```bash
git clone https://github.com/your-repo/mom-fresh.git
cd mom-fresh
python -m venv .venv
source .venv/bin/activate

# Install CUDA-enabled PyTorch (Adjust the index-url based on your CUDA version, e.g., cu118 or cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Configuration
Copy the `.env` file and configure it for GPU execution:

```bash
cp .env.example .env
```
Edit `.env`:
- Set `DEVICE=cuda`
- Set `COMPUTE_TYPE=float16` (Highly recommended for speed and memory efficiency on GPUs).
- Provide your `HF_TOKEN` and `GROQ_API_KEY`.

*(Optional)*: If you want to use a high-context local LLM (like Llama-3-70B) on the DGX to handle 2-hour long transcripts natively without Groq, you can implement a custom `LocalProvider` in `app/llm/provider.py` and point the `MoMGenerator` to it.

### 4. Running the Server
Start the server just like on CPU:
```bash
python start_server.py
```

With the GPU enabled, WhisperX transcription and Pyannote diarization will run significantly faster (expect < 5 seconds of processing time per 60 seconds of audio).

---

## 📁 API Endpoints

- `GET /health` : Health check.
- `POST /meetings/process` : Upload a `.wav`/`.mp3` file for full processing. Returns JSON transcript and MoM.
- `WS /meetings/stream` : WebSocket endpoint for live audio streaming. Accepts binary audio data and returns incremental JSON MoM updates every 60 seconds.
