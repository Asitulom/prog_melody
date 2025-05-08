#melody_model.py

#melody_model.py

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
import os

# Obtener la ruta de la carpeta actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de los archivos
JSON_PATH = os.path.join(BASE_DIR, "sad_midi_data.json")
NOTE_SCALER_PATH = os.path.join(BASE_DIR, "note_scaler.pkl")
DURATION_SCALER_PATH = os.path.join(BASE_DIR, "duration_scaler.pkl")
VELOCITY_SCALER_PATH = os.path.join(BASE_DIR, "velocity_scaler.pkl")
SCALE_ENCODER_PATH = os.path.join(BASE_DIR, "scale_encoder.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "melody_model.h5")

# Definición de escalas
SCALES = ["AMINOR", "A#MINOR", "BMINOR", "CMINOR", "C#MINOR", "DMINOR", "D#MINOR", "EMINOR", "FMINOR", "F#MINOR", "GMINOR", "G#MINOR"]

# Cargar datos de sad_midi_data.json
def load_midi_data():
    with open(JSON_PATH, "r") as file:
        data = json.load(file)
    melodies = []
    for scale, content in data.items():
        for note in content['melodia']:
            melodies.append({
                "note": note["note"],
                "duration": note["duration"],
                "velocity": note["velocity"],
                "scale": scale
            })
    return melodies

# Preprocesamiento de datos
def preprocess_data(melodies):
    notes, durations, velocities, scales = [], [], [], []

    for melody in melodies:
        notes.append(melody["note"])
        durations.append(melody["duration"])
        velocities.append(melody["velocity"])
        scales.append(melody["scale"])

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

    # Escalador para las escalas
    scale_encoder = OneHotEncoder()
    scales = scale_encoder.fit_transform(np.array(scales).reshape(-1, 1)).toarray()

    # Guardar escaladores
    with open(NOTE_SCALER_PATH, "wb") as f:
        pickle.dump(note_scaler, f)
    with open(DURATION_SCALER_PATH, "wb") as f:
        pickle.dump(duration_scaler, f)
    with open(VELOCITY_SCALER_PATH, "wb") as f:
        pickle.dump(velocity_scaler, f)
    with open(SCALE_ENCODER_PATH, "wb") as f:
        pickle.dump(scale_encoder, f)

    # Crear secuencias de entrenamiento
    sequence_length = 50
    X_notes, X_scales, y = [], [], []

    for i in range(len(notes) - sequence_length):
        X_notes.append(np.hstack((notes[i:i+sequence_length], durations[i:i+sequence_length], velocities[i:i+sequence_length])))
        X_scales.append(scales[i])  # La escala se mantiene constante
        y.append(notes[i + sequence_length])

    return np.array(X_notes), np.array(X_scales), np.array(y)

# Definir el modelo LSTM
def build_model(input_shape, scale_shape):
    input_notes = Input(shape=input_shape)
    input_scales = Input(shape=scale_shape)

    lstm_out = LSTM(128, return_sequences=True)(input_notes)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(128, return_sequences=False)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)

    merged = Concatenate()([lstm_out, input_scales])

    dense_out = Dense(64, activation='relu')(merged)
    output = Dense(1, activation='linear')(dense_out)

    model = tf.keras.Model(inputs=[input_notes, input_scales], outputs=output)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Entrenar el modelo
def train_model():
    melodies = load_midi_data()
    X_notes, X_scales, y = preprocess_data(melodies)

    model = build_model((X_notes.shape[1], X_notes.shape[2]), X_scales.shape[1])
    model.fit([X_notes, X_scales], y, epochs=100, batch_size=32, validation_split=0.2)

    # Guardar modelo
    model.save(MODEL_PATH)
    print(f"✅ Modelo entrenado y guardado como {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
