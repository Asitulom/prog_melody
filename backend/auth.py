#auth.py

import sqlite3
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

# Configuración del token JWT
SECRET_KEY = "clave-secreta-para-el-jwt"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Ruta a la base de datos
DB_PATH = "users.db"

# Crea contexto para hasheo de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 para autenticación
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Función para inicializar la base de datos
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

# Hashear contraseña
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Verificar contraseña
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# Crear token de acceso
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Obtener usuario desde el token
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token inválido")
        return username
    except Exception:
        raise HTTPException(status_code=401, detail="Token inválido")
