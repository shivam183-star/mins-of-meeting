import argparse
import json
import logging
from app.speech.pipeline import SpeechPipeline, format_llm_transcript

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Process audio through the Speech Pipeline.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("--json", type=str, default="transcript.json", help="Output JSON path")
    parser.add_argument("--text", type=str, default="transcript.txt", help="Output Text path")
    args = parser.parse_args()
    
    pipeline = SpeechPipeline()
    result = pipeline.process_audio(args.audio_file)
    
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
    llm_text = format_llm_transcript(result)
    with open(args.text, "w", encoding="utf-8") as f:
        f.write(llm_text)
        
    print(f"Successfully processed {args.audio_file}")
    print(f"Structured JSON: {args.json}")
    print(f"LLM Transcript: {args.text}")

if __name__ == "__main__":
    main()
