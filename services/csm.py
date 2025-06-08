from pathlib import Path
import os
import torchaudio
import torch
from .generator1 import load_csm_1b, Segment

class csm:
    def __init__(self, text):
        self.t = text

    def get_audio(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = load_csm_1b(device=device)

        speakers = [0]
        transcripts = [
            """Medical leave will be granted based on the individual situation and approved by Management. 
            Employees requesting medical leave must submit a valid medical certificate or doctor’s note within 3 working days 
            of the leave start date for consideration."""
        ]

        base_dir = Path(__file__).parent
        audio_paths = [base_dir / 'thuwa.wav']

        def load_audio(audio_path):
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_tensor, sample_rate = torchaudio.load(str(audio_path))
            if audio_tensor.dim() > 1 and audio_tensor.size(0) > 1:
                audio_tensor = audio_tensor.mean(dim=0)
            else:
                audio_tensor = audio_tensor.squeeze(0)
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate
            )
            return audio_tensor

        segments = [
            Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
            for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
        ]

        audio = generator.generate(
            text=self.t,
            speaker=0,
            context=segments,
            max_audio_length_ms=100000,
        )

        return audio
