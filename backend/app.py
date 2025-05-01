#app.py

from fastapi import FastAPI, Depends, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
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

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio donde se subirán las melodías de usuarios
UPLOAD_DIR = "ai/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --------- MODELOS Pydantic ---------
class MelodyRequest(BaseModel):
    tempo: int
    tone: str
    emotion: str

class ValoracionRequest(BaseModel):
    midi_name: str
    puntuacion: int

# --------- ENDPOINTS ---------

@app.post("/generate")
async def generate_melody(request: MelodyRequest):
    melody = generate_simple_melody(request.tempo, request.tone, request.emotion)
    with open("ai/generated_melody.json", "w") as f:
        json.dump(melody, f)
    return {"message": "Melodía generada correctamente"}

@app.get("/convert_to_midi/")
async def convert_to_midi():
    subprocess.run(["python", "ai/convert_to_midi.py"])
    midi_path = "ai/generated_melody.mid"
    if os.path.exists(midi_path):
        return FileResponse(midi_path, filename="generated_melody.mid", media_type="audio/midi")
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

@app.get("/melodias")
def listar_melodias():
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".mid")]
    return files

@app.get("/melodias/{filename}")
def descargar_melodia(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename, media_type="audio/midi")
    raise HTTPException(status_code=404, detail="Archivo no encontrado")

@app.post("/melodias/upload")
async def subir_melodia(file: UploadFile = File(...), username: str = Depends(get_current_user)):
    if not file.filename.endswith(".mid"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .mid")

    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)

    return JSONResponse(content={"message": f"Melodía {file.filename} subida por {username}."}, status_code=201)

@app.post("/valorar")
def valorar_melodia(valoracion: ValoracionRequest, username: str = Depends(get_current_user)):
    if not (1 <= valoracion.puntuacion <= 5):
        raise HTTPException(status_code=400, detail="La puntuación debe ser entre 1 y 5")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Comprobar si el usuario ya valoró esa melodía
    c.execute("""
        SELECT id FROM valoraciones
        WHERE midi_name = ? AND username = ?
    """, (valoracion.midi_name, username))
    ya_valoro = c.fetchone()

    if ya_valoro:
        conn.close()
        raise HTTPException(status_code=400, detail="Ya has valorado esta melodía")

    try:
        c.execute("""
            INSERT INTO valoraciones (midi_name, username, puntuacion)
            VALUES (?, ?, ?)
        """, (valoracion.midi_name, username, valoracion.puntuacion))
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar valoración: {str(e)}")
    finally:
        conn.close()

    return {"message": "Valoración registrada correctamente"}


@app.get("/valoraciones/{midi_name}")
def obtener_valoracion(midi_name: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT AVG(puntuacion), COUNT(*) FROM valoraciones WHERE midi_name = ?
    """, (midi_name,))
    result = c.fetchone()
    conn.close()

    if result and result[1] > 0:
        return {
            "midi_name": midi_name,
            "valoracion_media": round(result[0], 2),
            "cantidad_votos": result[1]
        }
    else:
        return {
            "midi_name": midi_name,
            "valoracion_media": None,
            "cantidad_votos": 0
        }
