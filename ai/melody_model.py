 
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pickle

# Cargar datos de sad_midi_data.json
def load_midi_data(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    melodies = []
    for melody_name in data:
        melodies.append(data[melody_name])

    return melodies

# Preprocesamiento de datos
def preprocess_data(melodies):
    note_sequences = []
    durations = []
    velocities = []

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
    note_scaler = MinMaxScaler()
    duration_scaler = MinMaxScaler()
    velocity_scaler = MinMaxScaler()

    note_sequences = note_scaler.fit_transform(note_sequences)
    durations = duration_scaler.fit_transform(durations)
    velocities = velocity_scaler.fit_transform(velocities)

    # Guardar escaladores para desnormalizar después
    with open("ai/note_scaler.pkl", "wb") as f:
        pickle.dump(note_scaler, f)
    with open("ai/duration_scaler.pkl", "wb") as f:
        pickle.dump(duration_scaler, f)
    with open("ai/velocity_scaler.pkl", "wb") as f:
        pickle.dump(velocity_scaler, f)

    # Crear secuencias de entrenamiento
    sequence_length = 50  # Tamaño de la ventana de entrenamiento
    X, y = [], []
    for i in range(len(note_sequences) - sequence_length):
        X.append(np.hstack((note_sequences[i:i+sequence_length], 
                            durations[i:i+sequence_length], 
                            velocities[i:i+sequence_length])))
        y.append(note_sequences[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    return X, y

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
    json_path = "ai/sad_midi_data.json"
    melodies = load_midi_data(json_path)
    X, y = preprocess_data(melodies)

    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    # Guardar modelo entrenado
    model.save("ai/melody_model.h5")
    print("✅ Modelo entrenado y guardado como melody_model.h5")

if __name__ == "__main__":
    train_model()
