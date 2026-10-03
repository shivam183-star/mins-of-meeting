import time
import logging

from app.config import config  # MUST BE IMPORTED FIRST to set HF_HOME!
import whisperx
from whisperx.diarize import DiarizationPipeline
from app.speech.schema import StructuredTranscript, MeetingMetadata, SpeakerTurn, Word, Speaker

logger = logging.getLogger(__name__)

class SpeechPipeline:
    def __init__(self):
        self.device = config.DEVICE
        self.compute_type = config.COMPUTE_TYPE
        
        logger.info(f"Loading ASR model: {config.ASR_MODEL} on {self.device}")
        # Using default asr_options, but passing specific vad_options to be slightly more lenient
        # to catch missing speech at the end of the file without being so loose it hallucinates.
        self.asr_model = whisperx.load_model(
            config.ASR_MODEL, 
            self.device, 
            compute_type=self.compute_type,
            vad_options={"vad_onset": 0.400, "vad_offset": 0.200}
        )
        
        self.diarize_model = None
        if config.ENABLE_DIARIZATION:
            if not config.HF_TOKEN:
                logger.warning("Diarization enabled but HF_TOKEN not found. Diarization will fail or be skipped.")
            else:
                logger.info(f"Loading Diarization model: {config.DIARIZATION_MODEL}...")
                self.diarize_model = DiarizationPipeline(
                    token=config.HF_TOKEN, 
                    device=self.device, 
                    cache_dir=config.MODEL_CACHE_DIR,
                    model_name=config.DIARIZATION_MODEL
                )

    def process_audio(self, audio_path: str) -> StructuredTranscript:
        start_time = time.time()
        logger.info(f"Loading audio: {audio_path}")
        
        # 1. Load Audio
        audio = whisperx.load_audio(audio_path)
        # Duration calculation based on WhisperX sample rate (16000)
        duration = len(audio) / 16000.0
        
        # 2. Transcription (Forcing language="en" so Hindi becomes Hinglish transliteration)
        logger.info("Transcribing audio with forced English for Hinglish transliteration...")
        asr_result = self.asr_model.transcribe(audio, batch_size=config.BATCH_SIZE, language="en")
        language = asr_result["language"]
        logger.info(f"Detected language: {language}")

        # 3. Alignment
        if config.ENABLE_ALIGNMENT:
            try:
                logger.info("Aligning transcript...")
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=language, 
                    device=self.device
                )
                asr_result = whisperx.align(
                    asr_result["segments"], 
                    align_model, 
                    align_metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=False
                )
            except Exception as e:
                logger.error(f"Alignment failed: {e}")
        
        # 4. Diarization
        if self.diarize_model:
            logger.info("Running diarization...")
            try:
                diarize_segments = self.diarize_model(audio)
                asr_result = whisperx.assign_word_speakers(diarize_segments, asr_result, fill_nearest=True)
            except Exception as e:
                logger.error(f"Diarization failed: {e}")
                
        # 5. Build Chronological Structured Transcript
        turns = self._build_turns(asr_result)
        
        unique_speakers = sorted(list(set([turn.speaker for turn in turns])))
        speakers = [Speaker(speaker_id=spk) for spk in unique_speakers]
        
        metadata = MeetingMetadata(duration=duration, language=language)
        
        logger.info(f"Processing complete in {time.time() - start_time:.2f}s")
        return StructuredTranscript(
            meeting_metadata=metadata,
            speakers=speakers,
            turns=turns
        )

    def _build_turns(self, asr_result: dict) -> list[SpeakerTurn]:
        """
        Reconstruct chronological speaker turns.
        If diarization wasn't run or failed, 'speaker' might be missing, so we default to 'SPEAKER_UNKNOWN'.
        """
        turns = []
        
        # WhisperX returns words in asr_result["word_segments"] if alignment was successful.
        # If alignment failed/skipped, we fall back to segments.
        has_words = "word_segments" in asr_result and len(asr_result["word_segments"]) > 0
        
        if has_words:
            current_turn_words = []
            current_speaker = None
            
            for word_info in asr_result["word_segments"]:
                # Sometimes start/end are missing if alignment couldn't align the word properly
                if "start" not in word_info or "end" not in word_info:
                    continue
                    
                spk = word_info.get("speaker", "SPEAKER_UNKNOWN")
                w = Word(
                    word=word_info["word"],
                    start=word_info["start"],
                    end=word_info["end"],
                    score=word_info.get("score"),
                    speaker=spk
                )
                
                if current_speaker is None:
                    current_speaker = spk
                    current_turn_words.append(w)
                elif spk == current_speaker:
                    # Check for large silence gap (e.g. > 2.0 seconds) to split same speaker turn
                    last_end = current_turn_words[-1].end
                    if w.start - last_end > 2.0:
                        # Finalize current turn
                        turns.append(self._create_turn(current_speaker, current_turn_words))
                        current_turn_words = [w]
                    else:
                        current_turn_words.append(w)
                else:
                    # Speaker changed!
                    turns.append(self._create_turn(current_speaker, current_turn_words))
                    current_speaker = spk
                    current_turn_words = [w]
            
            if current_turn_words:
                turns.append(self._create_turn(current_speaker, current_turn_words))
                
        else:
            # Fallback to segments if word-level alignment isn't available
            for seg in asr_result.get("segments", []):
                spk = seg.get("speaker", "SPEAKER_UNKNOWN")
                # Approximate word info using the segment itself
                w = Word(
                    word=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    score=None,
                    speaker=spk
                )
                turns.append(self._create_turn(spk, [w]))
                
        return turns
        
    def _create_turn(self, speaker: str, words: list[Word]) -> SpeakerTurn:
        start = words[0].start
        end = words[-1].end
        # Reconstruct text by joining words with a space
        text = " ".join([w.word for w in words]).strip()
        return SpeakerTurn(
            speaker=speaker,
            start=start,
            end=end,
            text=text,
            words=words
        )

def format_llm_transcript(transcript: StructuredTranscript) -> str:
    """Formats the structured transcript into a readable text format for LLMs."""
    lines = []
    for turn in transcript.turns:
        # Convert seconds to MM:SS format roughly
        start_min, start_sec = int(turn.start // 60), int(turn.start % 60)
        end_min, end_sec = int(turn.end // 60), int(turn.end % 60)
        
        time_str = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
        lines.append(f"{time_str} {turn.speaker}:\n{turn.text}\n")
    
    return "\n".join(lines)
