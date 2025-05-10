#generate_music.py

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from mido import MidiFile, MidiTrack, Message

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
CHORD_MODEL_PATH = os.path.join(BASE_DIR, '../ai/chord_model.h5')
MELODY_MODEL_PATH = os.path.join(BASE_DIR, '../ai/melody_model.h5')
NOTE_SCALER_PATH = os.path.join(BASE_DIR, '../ai/note_scaler.pkl')
DURATION_SCALER_PATH = os.path.join(BASE_DIR, '../ai/duration_scaler.pkl')
VELOCITY_SCALER_PATH = os.path.join(BASE_DIR, '../ai/velocity_scaler.pkl')

# SCALES
SCALES = {
    "AMINOR": ["A", "B", "C", "D", "E", "F", "G"],
    "A#MINOR": ["A#", "C", "C#", "D#", "F", "F#", "G#"],
    "BMINOR": ["B", "C#", "D", "E", "F#", "G", "A"],
    "CMINOR": ["C", "D", "Eb", "F", "G", "Ab", "Bb"],
    "C#MINOR": ["C#", "D#", "E", "F#", "G#", "A", "B"],
    "DMINOR": ["D", "E", "F", "G", "A", "Bb", "C"],
    "D#MINOR": ["D#", "F", "F#", "G#", "A#", "B", "C#"],
    "EMINOR": ["E", "F#", "G", "A", "B", "C", "D"],
    "FMINOR": ["F", "G", "Ab", "Bb", "C", "Db", "Eb"],
    "F#MINOR": ["F#", "G#", "A", "B", "C#", "D", "E"],
    "GMINOR": ["G", "A", "Bb", "C", "D", "Eb", "F"],
    "G#MINOR": ["G#", "A#", "B", "C#", "D#", "E", "F#"]
}

# Cargar modelos y escaladores
chord_model = tf.keras.models.load_model(CHORD_MODEL_PATH)
melody_model = tf.keras.models.load_model(MELODY_MODEL_PATH)

with open(NOTE_SCALER_PATH, 'rb') as f:
    note_scaler = pickle.load(f)

with open(DURATION_SCALER_PATH, 'rb') as f:
    duration_scaler = pickle.load(f)

with open(VELOCITY_SCALER_PATH, 'rb') as f:
    velocity_scaler = pickle.load(f)

# Función para verificar si una nota pertenece a la escala
def is_note_in_scale(note, scale):
    return note in SCALES.get(scale, [])

# Función para generar acordes
def generate_chords(scale, num_chords=4):
    chords = []
    input_sequence = np.zeros((1, 4, 6))

    for _ in range(num_chords):
        while True:
            predicted_chord = chord_model.predict(input_sequence)[0]
            notes_in_scale = [note for note in predicted_chord if is_note_in_scale(note, scale)]
            if len(notes_in_scale) == 4:
                chords.append(notes_in_scale)
                new_input = np.expand_dims(notes_in_scale, axis=0)
                input_sequence = np.concatenate([input_sequence[:, 1:, :], new_input.reshape(1, 1, -1)], axis=1)
                break

    return chords

# Función para generar melodía
def generate_melody(scale, length=8):
    melody = []
    input_sequence = np.zeros((1, 3, 3))

    for _ in range(length):
        while True:
            predicted_note = melody_model.predict(input_sequence)[0][0]
            if is_note_in_scale(predicted_note, scale):
                melody.append(predicted_note)
                new_input = np.expand_dims([predicted_note, 0.5, 0.5], axis=0)
                input_sequence = np.concatenate([input_sequence[:, 1:, :], new_input.reshape(1, 1, -1)], axis=1)
                break

    return melody

# Función para crear un archivo MIDI
def create_midi_file(chords, melody, output_path):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for chord in chords:
        for note in chord:
            track.append(Message('note_on', note=int(note), velocity=64, time=0))
        for note in chord:
            track.append(Message('note_off', note=int(note), velocity=64, time=480))

    for note in melody:
        track.append(Message('note_on', note=int(note), velocity=64, time=0))
        track.append(Message('note_off', note=int(note), velocity=64, time=480))

    mid.save(output_path)

# Función principal para generar música
def generate_music(scale, output_path='generated_music.mid'):
    chords = generate_chords(scale)
    melody = generate_melody(scale)
    create_midi_file(chords, melody, output_path)
    print(f'✅ Música generada y guardada en {output_path}')

if __name__ == '__main__':
    generate_music('AMINOR')
