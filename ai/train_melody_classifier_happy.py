#train_melody_classifier_happy.py
import os, json, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAPPY_JSON    = os.path.join(BASE_DIR, "happy_midi_data_augmented.json")
FULL_WEIGHTS  = os.path.join(BASE_DIR, "full_melody.h5")
NOTE_SCL_PKL  = os.path.join(BASE_DIR, "note_scaler_happy.pkl")
DUR_SCL_PKL   = os.path.join(BASE_DIR, "duration_scaler_happy.pkl")
VEL_SCL_PKL   = os.path.join(BASE_DIR, "velocity_scaler_happy.pkl")
OUTPUT_MODEL  = os.path.join(BASE_DIR, "melody_model_happy.h5")

def load_happy_mels():
    with open(HAPPY_JSON, "r") as f:
        raw = json.load(f)
    mels = []
    for block in raw.values():
        for m in block.get("melodias", []):
            mels.append((m["note"], m["duration"], m["velocity"]))
    print(f"🎯 Cargando {len(mels)} eventos de {os.path.basename(HAPPY_JSON)}")
    return mels


def preprocess(melodies):
    notes = np.array([m[0] for m in melodies]).reshape(-1,1).astype(float)
    durs  = np.array([m[1] for m in melodies]).reshape(-1,1).astype(float)
    vels  = np.array([m[2] for m in melodies]).reshape(-1,1).astype(float)

    note_scl = MinMaxScaler().fit(notes)
    dur_scl  = MinMaxScaler().fit(durs)
    vel_scl  = MinMaxScaler().fit(vels)

    # Guardar scalers para generación
    with open(NOTE_SCL_PKL, "wb") as f: pickle.dump(note_scl, f)
    with open(DUR_SCL_PKL,  "wb") as f: pickle.dump(dur_scl,  f)
    with open(VEL_SCL_PKL,  "wb") as f: pickle.dump(vel_scl,  f)

    n_s = note_scl.transform(notes)
    d_s = dur_scl.transform(durs)
    v_s = vel_scl.transform(vels)

    SEQ_LEN = 16
    X, y = [], []
    for i in range(len(n_s) - SEQ_LEN):
        seq = np.hstack((
            n_s[i:i+SEQ_LEN],
            d_s[i:i+SEQ_LEN],
            v_s[i:i+SEQ_LEN],
        ))
        X.append(seq)
        y.append([
            n_s[i+SEQ_LEN, 0],
            d_s[i+SEQ_LEN, 0],
            v_s[i+SEQ_LEN, 0]
        ])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"🔢 Fine-tune secuencias: {X.shape} -> {y.shape}")
    return X, y


def build_model(input_shape):
    m = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(3, activation="linear")
    ])
    m.compile(optimizer=Adam(1e-3), loss="mse")
    return m


def train():
    mels = load_happy_mels()
    X, y = preprocess(mels)
    if X.shape[0] < 10:
        print("⚠️ Pocos datos, no entreno.")
        return

    model = build_model((X.shape[1], X.shape[2]))
    print("🔄 Cargando pesos genéricos…")
    model.load_weights(FULL_WEIGHTS)

    print("🚀 Fine-tuning modelo en ‘happy’…")
    model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        shuffle=True
    )
    model.save(OUTPUT_MODEL)
    print(f"✅ Modelo fine-tuneado guardado en {OUTPUT_MODEL}")

if __name__ == "__main__":
    train()
