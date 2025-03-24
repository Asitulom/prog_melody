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


def generate_lstm_melody(sequence_length=50, num_notes=30):
    # Cargar modelo y escaladores
    model = load_model(MODEL_PATH)
    
    with open(NOTE_SCALER_PATH, "rb") as f:
        note_scaler = pickle.load(f)
    with open(DURATION_SCALER_PATH, "rb") as f:
        duration_scaler = pickle.load(f)
    with open(VELOCITY_SCALER_PATH, "rb") as f:
        velocity_scaler = pickle.load(f)

    # Semilla aleatoria para generar (puedes ajustar esto según tu dataset)
    # Por ahora creamos un array de valores aleatorios dentro del rango normalizado [0,1]
    seed = np.random.rand(sequence_length, 3)
    generated = []

    for _ in range(num_notes):
        X_input = np.array(seed).reshape(1, sequence_length, 3)
        pred_note = model.predict(X_input, verbose=0)[0]

        # Convertir la nota predicha a su valor original
        denorm_note = note_scaler.inverse_transform(pred_note.reshape(-1, 1))[0][0]

        # Duración y velocidad aleatorias dentro del rango normalizado (opcional: puedes entrenarlos también)
        rand_duration = duration_scaler.inverse_transform([[np.random.rand()]])[0][0]
        rand_velocity = velocity_scaler.inverse_transform([[np.random.rand()]])[0][0]

        generated.append({
            "note": int(round(denorm_note)),
            "duration": int(rand_duration),
            "velocity": int(rand_velocity)
        })

        # Actualiza la semilla desplazando la ventana
        seed = np.vstack([seed[1:], [[pred_note[0], np.random.rand(), np.random.rand()]]])

    return generated

def generate_simple_melody(tempo=120, tone="C", emotion="happy"):
    return generate_lstm_melody()