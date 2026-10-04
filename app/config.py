import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix caching globally for HuggingFace, faster-whisper, and Pyannote
os.environ["HF_HOME"] = os.getenv("MODEL_CACHE_DIR", os.path.join(os.getcwd(), ".model_cache"))

class Config:
    # ASR Settings
    ASR_MODEL = os.getenv("ASR_MODEL", "medium")
    DEVICE = os.getenv("DEVICE", "cpu")
    COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    
    # Cache Directory
    MODEL_CACHE_DIR = os.environ["HF_HOME"]
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # Diarization & Alignment
    HF_TOKEN = os.getenv("HF_TOKEN")
    ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
    ENABLE_ALIGNMENT = os.getenv("ENABLE_ALIGNMENT", "true").lower() == "true"
    DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")

    # LLM Settings
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq") # 'groq' or 'local'
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    
    # Local LLM Settings (for DGX Spark)
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "EMPTY")

config = Config()
