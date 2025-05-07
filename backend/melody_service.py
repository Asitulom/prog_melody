#melody_service.py

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Rutas
AI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ai"))
MODEL_PATH = os.path.join(AI_DIR, "melody_model.h5")
NOTE_SCALER_PATH = os.path.join(AI_DIR, "note_scaler.pkl")
DURATION_SCALER_PATH = os.path.join(AI_DIR, "duration_scaler.pkl")
VELOCITY_SCALER_PATH = os.path.join(AI_DIR, "velocity_scaler.pkl")
SCALE_ENCODER_PATH = os.path.join(AI_DIR, "scale_encoder.pkl")

# Mapear tonos a escalas menores
TONO_A_ESCALA = {
    "C": "CMINOR",
    "F": "FMINOR",
    "A#": "A#MINOR",
    "A": "AMINOR",
    "E": "EMINOR",
    "G": "GMINOR",
    "D#": "D#MINOR",
    "F#": "F#MINOR",
    "G#": "G#MINOR",
    "B": "BMINOR",
    "D": "DMINOR",
    "C#": "C#MINOR"
}

def generate_lstm_melody(tempo, tone, emotion, sequence_length=50, num_notes=30):
    # Cargar modelo y escaladores
    model = load_model(MODEL_PATH)
    with open(NOTE_SCALER_PATH, "rb") as f:
        note_scaler = pickle.load(f)
    with open(DURATION_SCALER_PATH, "rb") as f:
        duration_scaler = pickle.load(f)
    with open(VELOCITY_SCALER_PATH, "rb") as f:
        velocity_scaler = pickle.load(f)
    with open(SCALE_ENCODER_PATH, "rb") as f:
        scale_encoder = pickle.load(f)

    # Obtener la escala a partir del tono
    scale = TONO_A_ESCALA.get(tone, "CMINOR")
    scale_encoded = scale_encoder.transform([[scale]]).toarray()

    # Generar una semilla aleatoria
    seed = np.random.rand(sequence_length, 3)
    generated = []

    for _ in range(num_notes):
        X_input = np.array(seed).reshape(1, sequence_length, 3)
        pred_note = model.predict([X_input, scale_encoded], verbose=0)[0]

        # Desnormalizar la nota predicha
        denorm_note = note_scaler.inverse_transform(pred_note.reshape(-1, 1))[0][0]

        # Duración y velocidad aleatorias (opcional)
        rand_duration = duration_scaler.inverse_transform([[np.random.rand()]])[0][0]
        rand_velocity = velocity_scaler.inverse_transform([[np.random.rand()]])[0][0]

        # Ajustar los valores a rangos válidos
        note = max(0, min(127, int(round(denorm_note))))
        velocity = max(0, min(127, int(rand_velocity)))
        duration = max(0, min(1000, int(rand_duration)))

        generated.append({
            "note": note,
            "duration": duration,
            "velocity": velocity
        })

        # Actualizar la semilla con la nueva predicción
        seed = np.vstack([seed[1:], [[pred_note[0], np.random.rand(), np.random.rand()]]])

    return generated

def generate_simple_melody(tempo, tone, emotion):
    return generate_lstm_melody(tempo, tone, emotion)
