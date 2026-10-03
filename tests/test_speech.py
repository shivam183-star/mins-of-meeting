import unittest
from unittest.mock import patch, MagicMock
from app.speech.pipeline import SpeechPipeline
from app.speech.schema import StructuredTranscript

class TestSpeechPipeline(unittest.TestCase):
    
    @patch('whisperx.load_audio')
    @patch('whisperx.load_model')
    def test_pipeline_grouping(self, mock_load_model, mock_load_audio):
        # Mock audio and model
        mock_load_audio.return_value = [0]*160000  # 10 seconds of audio
        
        mock_model = MagicMock()
        
        # Mock ASR result with word_segments simulating chronological speaker turns
        mock_asr_result = {
            "language": "en",
            "segments": [],
            "word_segments": [
                {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                {"word": "world", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
                {"word": "Hi", "start": 1.5, "end": 2.0, "speaker": "SPEAKER_01"},
                {"word": "there", "start": 2.0, "end": 2.5, "speaker": "SPEAKER_01"},
                {"word": "Wait", "start": 3.0, "end": 3.5, "speaker": "SPEAKER_00"},
            ]
        }
        mock_model.transcribe.return_value = mock_asr_result
        mock_load_model.return_value = mock_model
        
        # Disable diarization and alignment dynamically for testing grouping logic
        with patch('app.config.config.ENABLE_DIARIZATION', False), \
             patch('app.config.config.ENABLE_ALIGNMENT', False):
            pipeline = SpeechPipeline()
            # Overwrite ASR model mock behavior since pipeline runs it
            
            transcript = pipeline.process_audio("dummy.wav")
            
            self.assertEqual(transcript.meeting_metadata.duration, 10.0)
            self.assertEqual(transcript.meeting_metadata.language, "en")
            
            # Should have 3 turns: 00 -> 01 -> 00
            self.assertEqual(len(transcript.turns), 3)
            self.assertEqual(transcript.turns[0].speaker, "SPEAKER_00")
            self.assertEqual(transcript.turns[0].text, "Hello world")
            self.assertEqual(transcript.turns[1].speaker, "SPEAKER_01")
            self.assertEqual(transcript.turns[1].text, "Hi there")
            self.assertEqual(transcript.turns[2].speaker, "SPEAKER_00")
            self.assertEqual(transcript.turns[2].text, "Wait")
            
            # Word timestamps preserved
            self.assertEqual(transcript.turns[0].words[0].start, 0.0)
            self.assertEqual(transcript.turns[0].words[-1].end, 1.0)
            
if __name__ == '__main__':
    unittest.main()
