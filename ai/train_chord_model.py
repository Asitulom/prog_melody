#train_chord_model

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
JSON_PATH = os.path.join(BASE_DIR, "sad_midi_data.json")
CHORD_MODEL_PATH = os.path.join(BASE_DIR, "chord_model.h5")

def load_chord_data():
    with open(JSON_PATH, "r") as file:
        data = json.load(file)

    chords = []

    for scale, content in data.items():
        for chord in content['acordes']:
            chords.append({
                "notes": chord["notes"],
                "duration": chord["duration"],
                "velocity": chord["velocity"],
                "scale": scale
            })

    print(f"Total de acordes cargados: {len(chords)}")
    return chords


def preprocess_chord_data(chords):
    X, y = [], []
    sequence_length = 4

    for i in range(len(chords) - sequence_length):
        input_sequence = chords[i:i + sequence_length]
        output_chord = chords[i + sequence_length]

        # Convertir los acordes a un array de características
        input_features = []
        for chord in input_sequence:
            # Asegurar que cada acorde tenga 4 notas
            notes = chord["notes"] + [0] * (4 - len(chord["notes"]))  # Padding
            input_features.append(notes + [chord["duration"], chord["velocity"]])

        # Convertir el acorde objetivo a la misma estructura
        output_notes = output_chord["notes"] + [0] * (4 - len(output_chord["notes"]))

        X.append(input_features)
        y.append(output_notes)

    X = np.array(X)
    y = np.array(y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y


def build_chord_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(4, activation='linear')  # 4 notas en el acorde
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_chord_model():
    chords = load_chord_data()
    X, y = preprocess_chord_data(chords)

    if X.shape[0] == 0:
        print("⚠️ No hay datos suficientes para entrenar el modelo de acordes.")
        return

    model = build_chord_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)

    model.save(CHORD_MODEL_PATH)
    print(f"✅ Modelo de acordes guardado en {CHORD_MODEL_PATH}")


if __name__ == "__main__":
    train_chord_model()
