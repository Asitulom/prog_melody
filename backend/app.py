#app.py

from fastapi import FastAPI, Depends, HTTPException, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
import subprocess
import os
import json  
import sqlite3
from backend.melody_service import generate_simple_melody
from backend.auth import (
    init_db, hash_password, verify_password, create_access_token,
    get_current_user, DB_PATH
)

app = FastAPI()

init_db()


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

@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    hashed = hash_password(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="El usuario ya existe")
    finally:
        conn.close()
    return {"message": f"Usuario {username} registrado correctamente"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (form_data.username,))
    result = c.fetchone()
    conn.close()

    if not result or not verify_password(form_data.password, result[0]):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    token = create_access_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}
