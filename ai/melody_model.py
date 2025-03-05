#melody_model.py

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Obtener la ruta de la carpeta actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de los archivos
JSON_PATH = os.path.join(BASE_DIR, "sad_midi_data.json")
NOTE_SCALER_PATH = os.path.join(BASE_DIR, "note_scaler.pkl")
DURATION_SCALER_PATH = os.path.join(BASE_DIR, "duration_scaler.pkl")
VELOCITY_SCALER_PATH = os.path.join(BASE_DIR, "velocity_scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "melody_model.h5")

# Cargar datos de sad_midi_data.json
def load_midi_data():
    with open(JSON_PATH, "r") as file:
        data = json.load(file)

    melodies = list(data.values())  # Extrae las melodías en una lista
    return melodies

# Preprocesamiento de datos
def preprocess_data(melodies):
    note_sequences, durations, velocities = [], [], []

    for melody in melodies:
        for note in melody:
            note_sequences.append(note["note"])
            durations.append(note["duration"])
            velocities.append(note["velocity"])

    # Convertir a arrays numpy
    note_sequences = np.array(note_sequences).reshape(-1, 1)
    durations = np.array(durations).reshape(-1, 1)
    velocities = np.array(velocities).reshape(-1, 1)

    # Normalización (0-1) con MinMaxScaler
    note_scaler, duration_scaler, velocity_scaler = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    
    note_sequences = note_scaler.fit_transform(note_sequences)
    durations = duration_scaler.fit_transform(durations)
    velocities = velocity_scaler.fit_transform(velocities)

    # Guardar escaladores en la misma carpeta que el script
    with open(NOTE_SCALER_PATH, "wb") as f:
        pickle.dump(note_scaler, f)
    with open(DURATION_SCALER_PATH, "wb") as f:
        pickle.dump(duration_scaler, f)
    with open(VELOCITY_SCALER_PATH, "wb") as f:
        pickle.dump(velocity_scaler, f)

    # Crear secuencias de entrenamiento
    sequence_length = 50  # Tamaño de la ventana de entrenamiento
    X, y = [], []
    for i in range(len(note_sequences) - sequence_length):
        X.append(np.hstack((note_sequences[i:i+sequence_length], 
                            durations[i:i+sequence_length], 
                            velocities[i:i+sequence_length])))
        y.append(note_sequences[i+sequence_length])

    return np.array(X), np.array(y)

# Definir el modelo LSTM
def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Salida normalizada de notas
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Entrenar el modelo
def train_model():
    melodies = load_midi_data()
    X, y = preprocess_data(melodies)

    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    # Guardar modelo entrenado en la misma carpeta que el script
    model.save(MODEL_PATH)
    print(f"✅ Modelo entrenado y guardado como {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
    