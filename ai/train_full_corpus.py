# ai/train_full_corpus.py

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from process_midi import process_midi_file  # función de extracción de melodías
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR  = os.path.join(BASE_DIR, "datasets", "corpus")
FULL_WEIGHTS= os.path.join(BASE_DIR, "full_melody.h5")
NOTE_SCL    = os.path.join(BASE_DIR, "full_note_scl.pkl")
DUR_SCL     = os.path.join(BASE_DIR, "full_dur_scl.pkl")
VEL_SCL     = os.path.join(BASE_DIR, "full_vel_scl.pkl")

# 1) Extraer melodías, saltando ficheros corruptos
print("📂 Leyendo corpus en:", CORPUS_DIR)
all_mels = []
for fn in os.listdir(CORPUS_DIR):
    if not fn.lower().endswith(".mid"):
        continue
    path = os.path.join(CORPUS_DIR, fn)
    try:
        mels = process_midi_file(path)
        if not mels:
            print(f"⚠️ {fn}: no extrajo notas (vacío), se ignora.")
        else:
            all_mels.extend(mels)
    except Exception as e:
        print(f"⚠️ {fn}: no es MIDI válido o error → {e}. Se ignora.")

if not all_mels:
    raise RuntimeError("❌ No se extrajo ninguna melodía válida. Revisa tu corpus.")

print(f"🎵 Eventos melódicos extraídos: {len(all_mels)}")

# 2) Construir arrays de note/duration/velocity
notes = np.array([m["note"]     for m in all_mels]).reshape(-1,1).astype(float)
durs  = np.array([m["duration"] for m in all_mels]).reshape(-1,1).astype(float)
vels  = np.array([m["velocity"] for m in all_mels]).reshape(-1,1).astype(float)

# 3) Escalar cada dimensión a [0,1]
note_scl = MinMaxScaler().fit(notes)
dur_scl  = MinMaxScaler().fit(durs)
vel_scl  = MinMaxScaler().fit(vels)
n_s = note_scl.transform(notes)
d_s = dur_scl .transform(durs)
v_s = vel_scl .transform(vels)

# 4) Crear secuencias de largo=16
SEQ_LEN = 16
X, y = [], []
for i in range(len(n_s) - SEQ_LEN):
    seq = np.hstack((
        n_s[i    : i+SEQ_LEN],
        d_s[i    : i+SEQ_LEN],
        v_s[i    : i+SEQ_LEN],
    ))
    X.append(seq)
    y.append([
        n_s[i+SEQ_LEN,0],
        d_s[i+SEQ_LEN,0],
        v_s[i+SEQ_LEN,0],
    ])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(f"🔢 Secuencias totales: {X.shape[0]} → cada una de {SEQ_LEN}×3")

# ——— SAMPLE RÁPIDO PARA TEST ———
SAMPLE_SIZE = 20000 #100000
if X.shape[0] > SAMPLE_SIZE:
    idx = np.random.choice(X.shape[0], SAMPLE_SIZE, replace=False)
    X = X[idx]
    y = y[idx]
    print(f"🔍 Usando muestra de {SAMPLE_SIZE} secuencias para entrenamiento rápido")
# ————————————————

# 5) Definir modelo LSTM genérico
def build_model(input_shape):
    m = Sequential([
        LSTM(256, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dense(3,   activation="linear")
    ])
    m.compile(optimizer=Adam(1e-3), loss="mse")
    return m

# 6) Entrenar y guardar
model = build_model((SEQ_LEN, 3))
print("🚀 Entrenando modelo genérico sobre muestra…")

early = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    X, y,
    epochs=3,             #50
    batch_size=64,
    validation_split=0.1,
    shuffle=True,
    callbacks=[early]
)

# Guardar pesos y escaladores
model.save(FULL_WEIGHTS)
with open(NOTE_SCL, "wb") as f: pickle.dump(note_scl, f)
with open(DUR_SCL,  "wb") as f: pickle.dump(dur_scl,  f)
with open(VEL_SCL,  "wb") as f: pickle.dump(vel_scl,  f)

print("✅ Pre-entrenamiento (sample) completado.")
print("   Pesos →", FULL_WEIGHTS)
print("   Scalers →", NOTE_SCL, DUR_SCL, VEL_SCL)
