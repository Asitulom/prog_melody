#generate_music.py

# backend/generate_music.py

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from mido import MidiFile, MidiTrack, Message
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths a modelos y datos
CHORD_MODEL_PATH     = os.path.join(BASE_DIR, '../ai/chord_model.h5')
MELODY_MODEL_PATH    = os.path.join(BASE_DIR, '../ai/melody_model.h5')
NOTE_SCALER_PATH     = os.path.join(BASE_DIR, '../ai/note_scaler.pkl')
DURATION_SCALER_PATH = os.path.join(BASE_DIR, '../ai/duration_scaler.pkl')
VELOCITY_SCALER_PATH = os.path.join(BASE_DIR, '../ai/velocity_scaler.pkl')
DATA_JSON_PATH       = os.path.join(BASE_DIR, '../ai/sad_midi_data.json')

# Rango base para las melodías y acordes (en semitonos MIDI)
LOWER_BOUND = 48   # C3
UPPER_BOUND = 84   # C6

# Definición de escalas
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

# Mapeo semitonos (incluyendo bemoles enharmónicos)
NOTE_TO_SEMITONE = {
    "C":0,
    "C#":1, "Db":1,
    "D":2,
    "D#":3, "Eb":3,
    "E":4,  "Fb":4,
    "F":5,  "E#":5,
    "F#":6, "Gb":6,
    "G":7,
    "G#":8, "Ab":8,
    "A":9,
    "A#":10,"Bb":10,
    "B":11, "Cb":11
}

# Precomputo semitonos por escala
SCALE_SEMITONES = {
    scale: [NOTE_TO_SEMITONE[n] for n in notes]
    for scale, notes in SCALES.items()
}

# Carga de modelos
chord_model  = tf.keras.models.load_model(CHORD_MODEL_PATH)
melody_model = tf.keras.models.load_model(MELODY_MODEL_PATH)

# Carga de escaladores (si se quisieran reutilizar)
with open(NOTE_SCALER_PATH, 'rb') as f:
    note_scaler = pickle.load(f)
with open(DURATION_SCALER_PATH, 'rb') as f:
    duration_scaler = pickle.load(f)
with open(VELOCITY_SCALER_PATH, 'rb') as f:
    velocity_scaler = pickle.load(f)


def semitone_offset(from_scale: str, to_scale: str) -> int:
    src = SCALE_SEMITONES[from_scale][0]
    tgt = SCALE_SEMITONES[to_scale][0]
    diff = (tgt - src) % 12
    return diff if diff <= 6 else diff - 12


def transpose_notes(notes: List[int], offset: int) -> List[int]:
    return [n + offset for n in notes]


def is_semitone_in_scale(midi_note: int, scale: str) -> bool:
    return (midi_note % 12) in SCALE_SEMITONES[scale]


def clamp_to_range(n: int, low: int = LOWER_BOUND, high: int = UPPER_BOUND) -> int:
    """Desplaza la nota por octavas completas hasta que quede en [low, high]."""
    while n < low:
        n += 12
    while n > high:
        n -= 12
    return n


def generate_chords(target_scale: str, num_chords: int = 4) -> Tuple[List[List[int]], List[int], List[int]]:
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)

    # Solo entries de escalas menores
    minor_entries = [
        (scale, chord)
        for scale, content in data.items()
        if scale.endswith("MINOR")
        for chord in content['acordes']
    ]

    np.random.shuffle(minor_entries)
    seed_entries = minor_entries[:4]

    # Transponer semillas
    seed = []
    for base_scale, chord in seed_entries:
        offset = semitone_offset(base_scale, target_scale)
        seed.append({
            'notes':    transpose_notes(chord['notes'], offset),
            'duration': chord['duration'],
            'velocity': chord['velocity']
        })

    # Preparar input para el modelo
    seq = []
    for c in seed:
        notes_padded = c['notes'] + [0] * (4 - len(c['notes']))
        seq.append(notes_padded + [c['duration'], c['velocity']])
    input_seq = np.array([seq], dtype=np.float32)

    # Generar acordes
    generated = []
    for _ in range(num_chords):
        pred = chord_model.predict(input_seq, verbose=0)[0]
        pred_int = np.round(pred).astype(int).tolist()

        # Snap a escala
        chord = []
        for n in pred_int:
            if (n % 12) in SCALE_SEMITONES[target_scale]:
                chord.append(n)
            else:
                base = n % 12
                nearest = min(SCALE_SEMITONES[target_scale], key=lambda s: abs(s - base))
                chord.append((n // 12) * 12 + nearest)
        generated.append(chord)

        # Actualizar input_seq
        last_dur = input_seq[0, -1, 4]
        last_vel = input_seq[0, -1, 5]
        new_feat = np.array([chord + [last_dur, last_vel]], dtype=np.float32).reshape(1, 1, 6)
        input_seq = np.concatenate([input_seq[:, 1:, :], new_feat], axis=1)

    # Usamos la última duración/velocity semilla para todos
    chord_durs = [seed[-1]['duration']] * num_chords
    chord_vels = [seed[-1]['velocity']] * num_chords
    return generated, chord_durs, chord_vels


def generate_melody(target_scale: str, length: int = 8) -> Tuple[List[int], List[int], List[int]]:
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)

    minor_notes = [
        (scale, note_dict)
        for scale, content in data.items()
        if scale.endswith("MINOR")
        for note_dict in content['melodias']
    ]

    np.random.shuffle(minor_notes)
    seed_notes = minor_notes[:3]

    seed = []
    for base_scale, m in seed_notes:
        offset = semitone_offset(base_scale, target_scale)
        seed.append({
            'note':     m['note'] + offset,
            'duration': m['duration'],
            'velocity': m['velocity']
        })

    # Input para el modelo
    seq = [[m['note'], m['duration'], m['velocity']] for m in seed]
    input_seq = np.array([seq], dtype=np.float32)

    melody = []
    for _ in range(length):
        pred = melody_model.predict(input_seq, verbose=0)[0][0]
        note_int = int(round(pred))

        # Snap a escala
        if is_semitone_in_scale(note_int, target_scale):
            final_note = note_int
        else:
            base = note_int % 12
            nearest = min(SCALE_SEMITONES[target_scale], key=lambda s: abs(s - base))
            final_note = (note_int // 12) * 12 + nearest

        melody.append(final_note)

        last_dur = input_seq[0, -1, 1]
        last_vel = input_seq[0, -1, 2]
        new_feat = np.array([[final_note, last_dur, last_vel]], dtype=np.float32).reshape(1, 1, 3)
        input_seq = np.concatenate([input_seq[:, 1:, :], new_feat], axis=1)

    # Usamos la última duración/velocity semilla para todos
    melody_durs = [seed[-1]['duration']] * length
    melody_vels = [seed[-1]['velocity']] * length
    return melody, melody_durs, melody_vels


def create_midi_file(chords: List[List[int]],
                     chord_durs: List[int],
                     chord_vels: List[int],
                     melody: List[int],
                     melody_durs: List[int],
                     melody_vels: List[int],
                     output_path: str = 'generated_music.mid'):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Escribir acordes
    for chord, dur, vel in zip(chords, chord_durs, chord_vels):
        for n in chord:
            note = clamp_to_range(n)
            track.append(Message('note_on',  note=note, velocity=vel, time=0))
        for i, n in enumerate(chord):
            note = clamp_to_range(n)
            offset = dur + i * 5
            track.append(Message('note_off', note=note, velocity=vel, time=offset))

    # Escribir melodía
    for n, dur, vel in zip(melody, melody_durs, melody_vels):
        note = clamp_to_range(n)
        track.append(Message('note_on',  note=note, velocity=vel, time=0))
        track.append(Message('note_off', note=note, velocity=vel, time=dur))

    mid.save(output_path)
    print(f'✅ Música generada y guardada en {output_path}')


def generate_music(scale: str, output_path: str = 'generated_music.mid'):
    target = scale.upper()
    if not target.endswith('MINOR') and not target.endswith('MAJOR'):
        target = f"{target}MINOR"

    chords, chord_durs, chord_vels = generate_chords(target)
    melody, melody_durs, melody_vels = generate_melody(target)
    create_midi_file(chords, chord_durs, chord_vels,
                     melody, melody_durs, melody_vels,
                     output_path)
