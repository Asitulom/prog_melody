# Dockerfile

FROM python:3.10-slim

# 1) Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copiar y pip install solo requirements primero (cacheo)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copiar todo el proyecto
COPY . .

# 4) Preprocesar / aumentar dataset y entrenar todos los modelos
RUN python ai/process_midi.py \
 && python ai/process_midi_happy.py \
 && python ai/augment_transpose.py \
 && python ai/augment_transpose_happy.py \
 && python ai/train_full_corpus.py \
 && python ai/train_melody_classifier.py \
 && python ai/train_melody_classifier_happy.py \
 && echo "✅ Preprocesamiento y entrenamiento completados"

# 5) Exponer y comando por defecto
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
