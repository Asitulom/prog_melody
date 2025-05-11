# train_melody_classifier.py

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
JSON_PATH = os.path.join(BASE_DIR, "sad_midi_data.json")
NOTE_SCALER_PATH = os.path.join(BASE_DIR, "note_scaler.pkl")
DURATION_SCALER_PATH = os.path.join(BASE_DIR, "duration_scaler.pkl")
VELOCITY_SCALER_PATH = os.path.join(BASE_DIR, "velocity_scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "melody_model.h5")

def load_melody_data():
    """ Cargar los datos de melodías desde el JSON """
    with open(JSON_PATH, "r") as file:
        data = json.load(file)

    melodies = []

    for scale, content in data.items():
        for note in content['melodias']:
            melodies.append({
                "note": note["note"],
                "duration": note["duration"],
                "velocity": note["velocity"],
                "scale": scale
            })

    print(f"Total melodías cargadas: {len(melodies)}")
    return melodies

def preprocess_melody_data(melodies):
    """ Preprocesar los datos de melodías para el modelo """
    notes, durations, velocities = [], [], []

    for melody in melodies:
        notes.append(melody["note"])
        durations.append(melody["duration"])
        velocities.append(melody["velocity"])

    # Convertir a arrays numpy
    notes = np.array(notes).reshape(-1, 1)
    durations = np.array(durations).reshape(-1, 1)
    velocities = np.array(velocities).reshape(-1, 1)

    # Normalización (0-1) con MinMaxScaler
    note_scaler = MinMaxScaler()
    duration_scaler = MinMaxScaler()
    velocity_scaler = MinMaxScaler()

    notes = note_scaler.fit_transform(notes)
    durations = duration_scaler.fit_transform(durations)
    velocities = velocity_scaler.fit_transform(velocities)

    # Guardar los escaladores
    with open(NOTE_SCALER_PATH, "wb") as f:
        pickle.dump(note_scaler, f)
    with open(DURATION_SCALER_PATH, "wb") as f:
        pickle.dump(duration_scaler, f)
    with open(VELOCITY_SCALER_PATH, "wb") as f:
        pickle.dump(velocity_scaler, f)

    # Creación de secuencias
    sequence_length = 3  # Se reduce para pruebas
    X, y = [], []

    for i in range(len(notes) - sequence_length):
        # Concatenar notas, duraciones y velocidades
        seq = np.hstack((
            notes[i:i + sequence_length],
            durations[i:i + sequence_length],
            velocities[i:i + sequence_length]
        ))
        X.append(seq)
        y.append(notes[i + sequence_length])  # La siguiente nota a predecir

    X = np.array(X)
    y = np.array(y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    if X.shape[0] == 0:
        print("⚠️ No hay datos suficientes para entrenar el modelo.")
    
    return X, y

def build_melody_model(input_shape):
    """ Crear el modelo LSTM para melodías """
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_melody_model():
    """ Función principal para entrenar el modelo de melodía """
    melodies = load_melody_data()
    X, y = preprocess_melody_data(melodies)

    if X.shape[0] == 0:
        print("⚠️ No se puede entrenar el modelo debido a la falta de secuencias.")
        return

    model = build_melody_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

    model.save(MODEL_PATH)
    print(f"✅ Modelo de melodía guardado en {MODEL_PATH}")

if __name__ == "__main__":
    train_melody_model()
