# backend/generate_music.py

import os
import json
import pickle
import random
import numpy as np
import tensorflow as tf
from mido import MidiFile, MidiTrack, Message
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas a modelos y datos
MELODY_MODEL_PATH    = os.path.join(BASE_DIR, '../ai/melody_model.h5')
NOTE_SCALER_PATH     = os.path.join(BASE_DIR, '../ai/note_scaler.pkl')
DURATION_SCALER_PATH = os.path.join(BASE_DIR, '../ai/duration_scaler.pkl')
VELOCITY_SCALER_PATH = os.path.join(BASE_DIR, '../ai/velocity_scaler.pkl')
DATA_JSON_PATH       = os.path.join(BASE_DIR, '../ai/sad_midi_data.json')
DEGREES_PATH         = os.path.join(BASE_DIR, '../ai/scale_degrees.json')

# Rango MIDI válido
LOWER_BOUND = 48   # C3
UPPER_BOUND = 84   # C6

# Ticks por negra
TICKS_PER_BEAT = 384

# Duraciones posibles (en negras / quarter-notes),
# cada patrón suma 16 (4 compases de 4/4)
DURATION_PATTERNS = [
    [4, 4, 4, 4],  # 1 barra cada acorde
    [2, 6, 2, 6],
    [6, 2, 6, 2],
    [3, 5, 3, 5],
    [5, 3, 5, 3],
]

# Progresiones (grados 0=I,1=II°, ...,6=VII)
PROGRESSIONS = [
    [0,3,4,0],  [0,4,5,3],  [0,6,5,6],
    [0,5,2,6],  [0,2,3,4],  [0,5,3,2],
    [0,2,6,3],  [0,4,3,5],  [0,5,4,3],
    [0,6,0,4],
]

# Carga grados de cada escala
with open(DEGREES_PATH, 'r') as f:
    SCALE_DEGREES = json.load(f)

# Mapeo nota→semitono
NOTE_TO_SEMITONE = {
    "C":0, "C#":1, "Db":1,
    "D":2, "D#":3, "Eb":3,
    "E":4, "Fb":4,
    "F":5, "E#":5,
    "F#":6, "Gb":6,
    "G":7, "G#":8, "Ab":8,
    "A":9, "A#":10,"Bb":10,
    "B":11,"Cb":11
}

def clamp_to_range(n: int) -> int:
    """Asegura que n quede entre LOWER_BOUND y UPPER_BOUND ajustando octavas."""
    while n < LOWER_BOUND: n += 12
    while n > UPPER_BOUND: n -= 12
    return n

def build_triad_chords(
    target_scale: str,
    length: int = 4
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Construye triadas 1-3-5 según una progresión aleatoria.
    Aplica uno de los DURATION_PATTERNS elegido al azar.
    """
    prog = random.choice(PROGRESSIONS)
    # recorta o repite la progresión para obtener 'length' acordes
    degrees = [ prog[i % len(prog)] for i in range(length) ]

    # elige patrón de duraciones (en negras) y lo convierte a ticks
    pattern = random.choice(DURATION_PATTERNS)
    chord_durs = [ d * TICKS_PER_BEAT for d in pattern ]

    chords, vels = [], []
    for deg in degrees:
        root_name  = SCALE_DEGREES[target_scale][deg]
        third_name = SCALE_DEGREES[target_scale][(deg+2)%7]
        fifth_name = SCALE_DEGREES[target_scale][(deg+4)%7]

        root  = NOTE_TO_SEMITONE[root_name]
        third = NOTE_TO_SEMITONE[third_name]
        fifth = NOTE_TO_SEMITONE[fifth_name]

        chords.append([
            clamp_to_range(root),
            clamp_to_range(third),
            clamp_to_range(fifth)
        ])
        vels.append(random.randint(63, 95))  # velocity aleatorio [50%–75%]

    return chords, chord_durs, vels

# ————————————————————————————————————————————————
# Carga modelo de melodía y escaladores
melody_model = tf.keras.models.load_model(MELODY_MODEL_PATH)
with open(NOTE_SCALER_PATH,     'rb') as f: note_scaler     = pickle.load(f)
with open(DURATION_SCALER_PATH, 'rb') as f: duration_scaler = pickle.load(f)
with open(VELOCITY_SCALER_PATH, 'rb') as f: velocity_scaler = pickle.load(f)

def is_semitone_in_scale(midi_note: int, scale: str) -> bool:
    sems = [ NOTE_TO_SEMITONE[n] for n in SCALE_DEGREES[scale] ]
    return (midi_note % 12) in sems

def semitone_offset(from_scale: str, to_scale: str) -> int:
    src = NOTE_TO_SEMITONE[SCALE_DEGREES[from_scale][0]]
    tgt = NOTE_TO_SEMITONE[SCALE_DEGREES[to_scale][0]]
    diff = (tgt - src) % 12
    return diff if diff <= 6 else diff - 12

def generate_melody(
    target_scale: str,
    length: int = 8
) -> Tuple[List[int], List[int], List[int]]:
    """Genera melodía vía LSTM y la ajusta a la escala dada."""
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)

    # 3 notas semilla aleatorias (solo minor scales)
    minor_notes = [
        (scale, note_dict)
        for scale, content in data.items() if scale.endswith("MINOR")
        for note_dict in content['melodias']
    ]
    random.shuffle(minor_notes)
    seed_notes = minor_notes[:3]

    seed = []
    for base_scale, m in seed_notes:
        offset = semitone_offset(base_scale, target_scale)
        seed.append({
            'note':     m['note'] + offset,
            'duration': m['duration'],
            'velocity': m['velocity']
        })

    seq = [[s['note'], s['duration'], s['velocity']] for s in seed]
    input_seq = np.array([seq], dtype=np.float32)

    melody, mdurs, mvels = [], [], []
    for _ in range(length):
        pred = melody_model.predict(input_seq, verbose=0)[0][0]
        note_int = int(round(pred))

        # Snap a escala
        if is_semitone_in_scale(note_int, target_scale):
            final_note = note_int
        else:
            base = note_int % 12
            sems = [ NOTE_TO_SEMITONE[n] for n in SCALE_DEGREES[target_scale] ]
            nearest = min(sems, key=lambda s: abs(s-base))
            final_note = (note_int//12)*12 + nearest

        melody.append(final_note)
        mdurs.append(seed[-1]['duration'])
        mvels.append(seed[-1]['velocity'])

        last_dur = input_seq[0,-1,1]
        last_vel = input_seq[0,-1,2]
        new_feat = np.array([[final_note, last_dur, last_vel]],
                            dtype=np.float32).reshape(1,1,3)
        input_seq = np.concatenate([input_seq[:,1:,:], new_feat], axis=1)

    return melody, mdurs, mvels

def create_midi_file(
    chords: List[List[int]],
    chord_durs: List[int],
    chord_vels: List[int],
    melody: List[int],
    mel_durs: List[int],
    mel_vels: List[int],
    output_path: str = 'generated_music.mid'
):
    """Ensambla y guarda el MIDI con acordes y melodía."""
    mid   = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)

    # --- Acordes ---
    for chord, dur, vel in zip(chords, chord_durs, chord_vels):
        for note in chord:
            vel = random.randint(63, 80)
            track.append(Message('note_on',
                                 note=clamp_to_range(note),
                                 velocity=vel,
                                 time=0))
        for i, note in enumerate(chord):
            t = dur if i==0 else 0
            track.append(Message('note_off',
                                 note=clamp_to_range(note),
                                 velocity=vel,
                                 time=t))

    # --- Melodía ---
    for n, dur, vel in zip(melody, mel_durs, mel_vels):
        vel = random.randint(63, 80)
        track.append(Message('note_on',
                             note=clamp_to_range(n),
                             velocity=vel,
                             time=0))
        track.append(Message('note_off',
                             note=clamp_to_range(n),
                             velocity=vel,
                             time=dur))

    mid.save(output_path)
    print(f'✅ Música generada y guardada en {output_path}')

def generate_music(
    scale: str,
    output_path: str = 'generated_music.mid'
):
    """
    scale: e.g. 'C', 'F#' + emoción → 'CMINOR' o 'F#MAJOR'.
    """
    target = scale.upper()
    if not target.endswith(('MINOR','MAJOR')):
        target = f"{target}MINOR"

    chords, cdurs, cvels = build_triad_chords(target)
    melody, mdurs, mvels = generate_melody(target)
    create_midi_file(chords, cdurs, cvels, melody, mdurs, mvels, output_path)