# Dockerfile

FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Preprocesar MIDI y entrenar modelo al construir
RUN python ai/process_midi.py && python ai/melody_model.py

# Exponer puerto FastAPI
EXPOSE 8000

# Comando para iniciar el servidor
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
