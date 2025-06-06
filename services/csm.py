from .generator1 import load_csm_1b
from .generator1 import Segment
import torchaudio
import torch


class csm:
    def __init__(self,text):
        self.t= text
        
        
               
    def get_audio(self):    
        
       
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
             
        generator = load_csm_1b(device=device)
       
        speakers = [0
                    # 0,
                    # 0,
                
                ]

        transcripts = [
        """Medical leave will be granted based on the individual situation and approved by Management. 
           Employees requesting medical leave must submit a valid medical certificate or doctor’s note within 3 working days 
           of the leave start date for consideration.
        """
        
        #    """ Sri Lanka is a small island country in South Asia, full of beautiful beaches, green mountains, and wildlife. 
        #    It has a long history with old cities like Anuradhapura and famous places like Sigiriya Rock and the Temple of the Tooth.
        #    People in Sri Lanka are friendly, and the food is tasty with dishes like rice and curry, hoppers, and kottu.
        #    With its nature, culture, and kind people, Sri Lanka is a special place to visit.""" ,
        
        #    """ The ESP32 is a small and powerful device used to build smart electronics. It has built-in Wi-Fi and Bluetooth, 
        #    so it can connect to the internet and other devices easily. People use it in projects like home automation, smart sensors, and robots.
        #    It's popular because it's cheap, fast, and easy to use with Arduino code. """,
        
                    ]
        
        audio_paths = [
            "thuwa.wav",
            # "data/lakshan2.wav",
            # "data/lakshan3.wav",
        

        ]
                
        
        
        def load_audio(audio_path):
                audio_tensor, sample_rate = torchaudio.load(audio_path)
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
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
        

        return(audio)

        #torchaudio.save("audio_4_local.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)