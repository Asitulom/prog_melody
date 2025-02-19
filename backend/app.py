#app.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import subprocess
import os
import json  

from backend.melody_service import generate_simple_melody

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
    """Genera una melodía en JSON y la guarda en un archivo."""
    melody = generate_simple_melody(request.tempo, request.tone, request.emotion)

    # Guardar la melodía generada en un archivo JSON
    json_path = "ai/generated_melody.json"
    with open(json_path, "w") as file:
        json.dump(melody, file)  

    return {"message": "Melodía generada", "json_file": json_path}

@app.get("/convert_to_midi/")
async def convert_to_midi():
    """Convierte el JSON generado en un archivo MIDI y lo devuelve."""
    subprocess.run(["python", "ai/convert_to_midi.py"])

    midi_path = "ai/generated_melody.mid"
    
    if os.path.exists(midi_path):
        return FileResponse(midi_path, filename="generated_melody.mid", media_type="audio/midi")
    else:
        return {"error": "El archivo MIDI no se generó correctamente"}
