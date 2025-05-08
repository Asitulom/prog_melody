# melody_service.py

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

# Definición de escalas menores
SCALES = {
    "AMINOR": ["A", "B", "C", "D", "E", "F", "G"],
    "A#MINOR": ["A#", "C", "C#", "D#", "F", "F#", "G#"],
    "BMINOR": ["B", "C#", "D", "E", "F#", "G", "A"],
    "CMINOR": ["C", "D", "D#", "F", "G", "G#", "A#"],
    "C#MINOR": ["C#", "D#", "E", "F#", "G#", "A", "B"],
    "DMINOR": ["D", "E", "F", "G", "A", "A#", "C"],
    "D#MINOR": ["D#", "F", "F#", "G#", "A#", "B", "C#"],
    "EMINOR": ["E", "F#", "G", "A", "B", "C", "D"],
    "FMINOR": ["F", "G", "G#", "A#", "C", "C#", "D#"],
    "F#MINOR": ["F#", "G#", "A", "B", "C#", "D", "E"],
    "GMINOR": ["G", "A", "A#", "C", "D", "D#", "F"],
    "G#MINOR": ["G#", "A#", "B", "C#", "D#", "E", "F#"]
}

# Convertir bemoles a sostenidos
FLAT_TO_SHARP = {
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "Db": "C#",
    "Eb": "D#",
    "Fb": "E",
    "Gb": "F#"
}

MIDI_TO_NOTE = {i: v for i, v in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] * 11)}
NOTE_TO_MIDI = {v: k for k, v in MIDI_TO_NOTE.items()}

def convert_flat_to_sharp(note):
    """ Convierte notas bemoles a sus equivalentes en sostenidos """
    return FLAT_TO_SHARP.get(note, note)

def convert_scale_to_sharp(scale_notes):
    """ Convierte una lista de notas a sus equivalentes en sostenidos """
    return [convert_flat_to_sharp(note) for note in scale_notes]

# Generar melodía LSTM
def generate_lstm_melody(tempo, tone, emotion, sequence_length=50, num_notes=30):
    model = load_model(MODEL_PATH)

    with open(NOTE_SCALER_PATH, "rb") as f:
        note_scaler = pickle.load(f)
    with open(DURATION_SCALER_PATH, "rb") as f:
        duration_scaler = pickle.load(f)
    with open(VELOCITY_SCALER_PATH, "rb") as f:
        velocity_scaler = pickle.load(f)
    with open(SCALE_ENCODER_PATH, "rb") as f:
        scale_encoder = pickle.load(f)

    scale = TONO_A_ESCALA.get(tone, "CMINOR")
    scale_encoded = scale_encoder.transform([[scale]]).toarray()
    
    # Convertir escala a sostenidos
    scale_notes = convert_scale_to_sharp(SCALES[scale])
    scale_midi = [NOTE_TO_MIDI[n] for n in scale_notes]

    seed = np.random.rand(sequence_length, 3)
    generated = []

    for _ in range(num_notes):
        X_input = np.array(seed).reshape(1, sequence_length, 3)
        pred_note = model.predict([X_input, scale_encoded], verbose=0)[0]
        denorm_note = note_scaler.inverse_transform(pred_note.reshape(-1, 1))[0][0]

        # Filtrar notas dentro de la escala
        note_candidates = [n for n in scale_midi if abs(n - denorm_note) < 12]
        if note_candidates:
            note = min(note_candidates, key=lambda x: abs(x - denorm_note))
        else:
            note = int(np.random.choice(scale_midi))

        # Convertir a int para JSON
        note = int(note)
        rand_duration = int(duration_scaler.inverse_transform([[np.random.rand()]])[0][0])
        rand_velocity = int(velocity_scaler.inverse_transform([[np.random.rand()]])[0][0])

        generated.append({
            "note": note,
            "duration": max(0, min(1000, rand_duration)),
            "velocity": max(0, min(127, rand_velocity))
        })

        # Actualizar la semilla
        seed = np.vstack([seed[1:], [[note / 127, np.random.rand(), np.random.rand()]]])

    return generated

# Generar melodía simple
def generate_simple_melody(tempo, tone, emotion):
    return generate_lstm_melody(tempo, tone, emotion)
