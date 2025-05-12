# Dockerfile

FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Preprocesar MIDI y entrenar modelo de melodía
RUN python ai/process_midi.py \
    && python ai/train_melody_classifier.py \
    && echo "✅ Modelos de melodía entrenados"

# Entrenar también modelo de acordes
# RUN python ai/train_chord_model.py

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
