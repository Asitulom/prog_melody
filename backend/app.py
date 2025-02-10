from fastapi import FastAPI
from pydantic import BaseModel
from backend.melody_service import generate_simple_melody
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen (en desarrollo)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (POST, GET, OPTIONS)
    allow_headers=["*"],  # Permitir cualquier cabecera
)

class MelodyRequest(BaseModel):
    tempo: int
    tone: str
    emotion: str

@app.post("/generate")
async def generate_melody(request: MelodyRequest):
    melody = generate_simple_melody(request.tempo, request.tone, request.emotion)
    return {"melody": melody}
