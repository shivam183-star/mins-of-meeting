import argparse
import json
import os
import time

import whisperx
from whisperx.diarize import DiarizationPipeline


def main():
    parser = argparse.ArgumentParser(description="Minimal validation script for WhisperX pipeline.")
    parser.add_argument("audio_file", type=str, help="Path to the audio file to process.")
    parser.add_argument("--output", type=str, default="output.json", help="Path to save the resulting JSON.")
    parser.add_argument("--model", type=str, default="tiny", help="Whisper model to use (e.g., tiny, small).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda).")
    parser.add_argument("--compute_type", type=str, default="int8", help="Compute type (int8 or float32 for CPU).")
    
    args = parser.parse_args()

    audio_file = args.audio_file
    output_file = args.output
    device = args.device
    compute_type = args.compute_type
    model_name = args.model

    # Check for Hugging Face token for Pyannote Diarization
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN environment variable is not set.")
        print("Diarization requires a Hugging Face token to access gated Pyannote models.")
        print("Set it using: export HF_TOKEN='your_token'")
        print("Proceeding without diarization...")

    print(f"\n--- Starting WhisperX Validation Pipeline ---")
    print(f"Audio File: {audio_file}")
    print(f"Device: {device}, Compute Type: {compute_type}, Model: {model_name}\n")

    total_start_time = time.time()

    # 1. Load audio
    print("[1/5] Loading audio...")
    audio_load_start = time.time()
    audio = whisperx.load_audio(audio_file)
    print(f"      Audio loaded in {time.time() - audio_load_start:.2f} seconds.")

    # 2. Transcription (ASR)
    print(f"[2/5] Loading ASR model '{model_name}'...")
    asr_load_start = time.time()
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    print(f"      ASR model loaded in {time.time() - asr_load_start:.2f} seconds.")

    print("      Transcribing...")
    transcribe_start = time.time()
    result = model.transcribe(audio, batch_size=4) # Default batch size for fast inference
    detected_language = result["language"]
    print(f"      Transcription finished in {time.time() - transcribe_start:.2f} seconds.")
    print(f"      Detected Language: {detected_language}")

    # 3. Alignment
    print(f"[3/5] Loading Alignment model for '{detected_language}'...")
    try:
        align_load_start = time.time()
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        print(f"      Alignment model loaded in {time.time() - align_load_start:.2f} seconds.")

        print("      Aligning transcript with audio...")
        align_start = time.time()
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        print(f"      Alignment finished in {time.time() - align_start:.2f} seconds.")
    except Exception as e:
        print(f"      WARNING: Alignment failed or unsupported for language '{detected_language}'. Error: {e}")
        print("      Proceeding with unaligned transcript...")

    # 4. Diarization
    if hf_token:
        print("[4/5] Loading Diarization model...")
        diarize_load_start = time.time()
        try:
            diarize_model = DiarizationPipeline(token=hf_token, device=device)
            print(f"      Diarization model loaded in {time.time() - diarize_load_start:.2f} seconds.")

            print("      Performing Diarization...")
            diarize_start = time.time()
            diarize_segments = diarize_model(audio)
            print(f"      Diarization finished in {time.time() - diarize_start:.2f} seconds.")

            # 5. Assign Speakers
            print("[5/5] Assigning speakers to transcript...")
            assign_start = time.time()
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"      Speaker assignment finished in {time.time() - assign_start:.2f} seconds.")
        except Exception as e:
            print(f"      WARNING: Diarization failed. Error: {e}")
    else:
        print("[4/5] Skipping Diarization (No HF_TOKEN provided).")
        print("[5/5] Skipping Speaker Assignment.")

    # Save Results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total_time = time.time() - total_start_time
    print(f"Pipeline completed successfully in {total_time:.2f} seconds!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()
