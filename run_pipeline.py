import argparse
import json
import logging
import os
from app.speech.pipeline import SpeechPipeline, format_llm_transcript
from app.llm.generator import MoMGenerator
from app.llm.groq_provider import GroqProvider
from app.config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_llm_provider():
    if config.LLM_PROVIDER == "groq":
        return GroqProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")

def main():
    parser = argparse.ArgumentParser(description="Full AI Meeting Intelligence Pipeline (ASR + Diarization + MoM)")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--out_dir", default="outputs", help="Directory to save the generated artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.audio_file))[0]
    
    out_json = os.path.join(args.out_dir, f"{base_name}_transcript.json")
    out_txt = os.path.join(args.out_dir, f"{base_name}_transcript.txt")
    out_mom = os.path.join(args.out_dir, f"{base_name}_mom.json")

    # Step 1: Speech Pipeline
    logger.info("=== STEP 1: SPEECH PIPELINE ===")
    pipeline = SpeechPipeline()
    transcript = pipeline.process_audio(args.audio_file)
    
    # Save Speech JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, indent=2, ensure_ascii=False)
        
    # Save LLM-ready Text
    llm_text = format_llm_transcript(transcript)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(llm_text)
    
    logger.info(f"Saved transcript JSON to {out_json}")
    logger.info(f"Saved speaker-aware text to {out_txt}")

    # Step 2: LLM MoM Generation
    logger.info("=== STEP 2: LLM MOM GENERATION ===")
    provider = get_llm_provider()
    generator = MoMGenerator(provider)
    
    mom = generator.generate(llm_text)
    
    with open(out_mom, "w", encoding="utf-8") as f:
        f.write(mom.model_dump_json(indent=2))
        
    logger.info(f"Saved Minutes of Meeting to {out_mom}")
    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
