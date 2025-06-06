from services.csm import csm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
import io
import torchaudio
import uvicorn


app = FastAPI()

# Request schema
class TextRequest(BaseModel):
    text: str

@app.post("/generate-audio")
def generate_audio_wav(request: TextRequest):
    try:
        # Generate audio
        synth = csm(request.text)
        audio_tensor = synth.get_audio()

        # Save to BytesIO as WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor.unsqueeze(0).cpu(), 24000, format="wav")
        buffer.seek(0)

        # Return as audio/wav
        return Response(content=buffer.read(), media_type="audio/wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000) 