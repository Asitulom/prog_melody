# app.py

from fastapi import FastAPI, Depends, HTTPException, Form, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import os
import sqlite3
import shutil
import uuid
from jose import JWTError, jwt

from backend.generate_music import generate_music
from backend.auth import (
    init_db, hash_password, verify_password, create_access_token,
    get_current_user, DB_PATH, SECRET_KEY, ALGORITHM
)

app = FastAPI()
init_db()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "ai/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MIS_MELODIAS_DIR = "ai/mis_melodias"
os.makedirs(MIS_MELODIAS_DIR, exist_ok=True)


# --------- MODELOS ---------
class MelodyRequest(BaseModel):
    tone: str
    emotion: str


class ValoracionRequest(BaseModel):
    midi_name: str
    puntuacion: int


# --------- GENERAR MELODÍA ---------
@app.post("/generate")
async def generate_melody(request_data: MelodyRequest, request: Request):
    # extraer usuario (si hay)
    token = request.headers.get("Authorization")
    username = None
    if token and token.startswith("Bearer "):
        try:
            payload = jwt.decode(token.split()[1], SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
        except JWTError:
            pass

    # nombre temporal
    tmp_name = f"tmp_{uuid.uuid4().hex[:8]}.mid"
    tmp_path = os.path.join(MIS_MELODIAS_DIR, tmp_name)

    # construir escala
    base_tone = request_data.tone.strip().upper()
    emotion = request_data.emotion.strip().lower()
    if emotion == "sad":
        scale_key = f"{base_tone}MINOR"
    elif emotion == "happy":
        scale_key = f"{base_tone}MAJOR"
    else:
        scale_key = f"{base_tone}MINOR"

    # generar música
    try:
        generate_music(scale_key, output_path=tmp_path)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"No se pudo generar música: {e}")

    # nombre definitivo
    if username:
        final_name = f"{username}_{uuid.uuid4().hex[:8]}.mid"
    else:
        final_name = f"melodia_{uuid.uuid4().hex[:8]}.mid"
    final_path = os.path.join(MIS_MELODIAS_DIR, final_name)
    shutil.move(tmp_path, final_path)

    # guardar en BD
    if username:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO mis_melodias(username, midi_name) VALUES(?, ?)",
            (username, final_name)
        )
        conn.commit()
        conn.close()

    return FileResponse(final_path, filename=final_name, media_type="audio/midi")


# --------- AUTENTICACIÓN & USUARIOS ---------
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


# --------- MELODÍAS PÚBLICAS ---------
@app.get("/melodias")
def listar_melodias():
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".mid")]


@app.get("/melodias/{filename}")
def descargar_melodia(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename, media_type="audio/midi")
    raise HTTPException(status_code=404, detail="Archivo no encontrado")


# --------- SUBIR MELODÍA PÚBLICA ---------
@app.post("/melodias/upload")
async def subir_melodia(file: UploadFile = File(...), username: str = Depends(get_current_user)):
    if not file.filename.endswith(".mid"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .mid")
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    return JSONResponse({"message": f"Melodía {file.filename} subida por {username}"}, status_code=201)


# --------- VALORACIONES ---------
@app.post("/valorar")
def valorar_melodia(valoracion: ValoracionRequest, username: str = Depends(get_current_user)):
    if not (1 <= valoracion.puntuacion <= 5):
        raise HTTPException(status_code=400, detail="La puntuación debe ser entre 1 y 5")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM valoraciones WHERE midi_name = ? AND username = ?", (valoracion.midi_name, username))
    if c.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Ya has valorado esta melodía")
    c.execute("INSERT INTO valoraciones (midi_name, username, puntuacion) VALUES (?, ?, ?)",
              (valoracion.midi_name, username, valoracion.puntuacion))
    conn.commit()
    conn.close()
    return {"message": "Valoración registrada correctamente"}


@app.get("/valoraciones/{midi_name}")
def obtener_valoracion(midi_name: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT AVG(puntuacion), COUNT(*) FROM valoraciones WHERE midi_name = ?", (midi_name,))
    avg, cnt = c.fetchone()
    conn.close()
    return {
        "midi_name": midi_name,
        "valoracion_media": round(avg, 2) if cnt else None,
        "cantidad_votos": cnt
    }


# --------- MIS MELODÍAS ---------
@app.get("/mis-melodias")
def obtener_mis_melodias(username: str = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT midi_name FROM mis_melodias WHERE username = ?", (username,))
    rows = [r[0] for r in c.fetchall()]
    conn.close()
    # solo archivos que aún existan
    return [m for m in rows if os.path.exists(os.path.join(MIS_MELODIAS_DIR, m))]


@app.get("/mis-melodias/{filename}")
def descargar_melodia_personal(filename: str, username: str = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM mis_melodias WHERE username = ? AND midi_name = ?", (username, filename))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=403, detail="No tienes permiso para esta melodía")
    conn.close()
    path = os.path.join(MIS_MELODIAS_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename, media_type="audio/midi")
    raise HTTPException(status_code=404, detail="Archivo no encontrado")
