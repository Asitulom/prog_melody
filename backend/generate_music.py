# backend/generate_music.py

import os
import json
import pickle
import random
import numpy as np
import tensorflow as tf
from mido import MidiFile, MidiTrack, Message
from typing import List, Tuple

# ————————————————————————————————————————————————
# Paths
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MELODY_MODEL_PATH   = os.path.join(BASE_DIR, '../ai/melody_model.h5')
NOTE_SCALER_PATH    = os.path.join(BASE_DIR, '../ai/note_scaler.pkl')
DURATION_SCALER_PATH= os.path.join(BASE_DIR, '../ai/duration_scaler.pkl')
VELOCITY_SCALER_PATH= os.path.join(BASE_DIR, '../ai/velocity_scaler.pkl')
DATA_JSON_PATH      = os.path.join(BASE_DIR, '../ai/sad_midi_data.json')
DEGREES_PATH        = os.path.join(BASE_DIR, '../ai/scale_degrees.json')

# MIDI configuration
LOWER_BOUND     = 48   # C3
UPPER_BOUND     = 84   # C6
TICKS_PER_BEAT  = 384

# Duration patterns (in quarter-notes, sum to 16)
DURATION_PATTERNS = [
    [4,4,4,4],
    [2,6,2,6],
    [6,2,6,2],
]

# Chord progressions (degrees 0=I,1=II, …,6=VII)
PROGRESSIONS = [
    [0,3,4,0], [0,4,5,3], [0,6,5,6],
    [0,5,2,6], [0,2,3,4], [0,5,3,2],
    [0,2,6,3], [0,4,3,5], [0,5,4,3],
    [0,6,0,4],
]

# Load scale degrees
with open(DEGREES_PATH, 'r') as f:
    SCALE_DEGREES = json.load(f)

# Note name → semitone
NOTE_TO_SEMITONE = {
    "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3,
    "E":4, "Fb":4, "F":5, "E#":5, "F#":6, "Gb":6,
    "G":7, "G#":8, "Ab":8, "A":9, "A#":10,"Bb":10,
    "B":11,"Cb":11
}

# ————————————————————————————————————————————————
def clamp_to_range(n: int) -> int:
    while n < LOWER_BOUND: n += 12
    while n > UPPER_BOUND: n -= 12
    return n

def is_semitone_in_scale(midi_note: int, scale: str) -> bool:
    sems = [NOTE_TO_SEMITONE[n] for n in SCALE_DEGREES[scale]]
    return (midi_note % 12) in sems

def semitone_offset(from_scale: str, to_scale: str) -> int:
    src = NOTE_TO_SEMITONE[SCALE_DEGREES[from_scale][0]]
    tgt = NOTE_TO_SEMITONE[SCALE_DEGREES[to_scale][0]]
    diff = (tgt - src) % 12
    return diff if diff <= 6 else diff - 12

# ————————————————————————————————————————————————
def build_triad_chords(
    target_scale: str,
    length: int = 4
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Genera 'length' acordes donde el 50% son triadas y el 50% llevan séptima.
    """
    prog       = random.choice(PROGRESSIONS)
    degrees    = [prog[i % len(prog)] for i in range(length)]
    pattern    = random.choice(DURATION_PATTERNS)
    chord_durs = [d * TICKS_PER_BEAT for d in pattern]

    # Solo dos opciones: triad o 7th, con pesos 0.5 y 0.5
    ext_types = ['triad', '7th']
    weights   = [0.5,     0.5]

    # Elegimos 'length' banderas según esos pesos
    ext_flags = random.choices(ext_types, weights=weights, k=length)

    chords, vels = [], []
    notes        = SCALE_DEGREES[target_scale]

    for deg, ext in zip(degrees, ext_flags):
        # Construir la triada básica
        root  = notes[deg]
        third = notes[(deg+2)%7]
        fifth = notes[(deg+4)%7]
        chord_names = [root, third, fifth]

        # Si toca séptima, añadimos grado VII
        if ext == '7th':
            chord_names.append(notes[(deg+6)%7])

        # Convertir nombres a semitonos y ajustar al rango
        semis = [ clamp_to_range(NOTE_TO_SEMITONE[n]) for n in chord_names ]
        chords.append(semis)

        # Velocidad aleatoria
        vels.append(random.randint(63, 95))

    return chords, chord_durs, vels


# ————————————————————————————————————————————————
# Load melody model & scalers
melody_model = tf.keras.models.load_model(MELODY_MODEL_PATH)
with open(NOTE_SCALER_PATH,     'rb') as f: note_scaler     = pickle.load(f)
with open(DURATION_SCALER_PATH, 'rb') as f: duration_scaler = pickle.load(f)
with open(VELOCITY_SCALER_PATH, 'rb') as f: velocity_scaler = pickle.load(f)

def generate_melody(
    target_scale: str,
    length: int = 8
) -> Tuple[List[int], List[int], List[int]]:
    """Genera melodía vía LSTM y la ajusta a la escala dada, usando siempre 16 pasos de contexto."""
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)

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

    # seq shape = (3,3)
    seq = [[s['note'], s['duration'], s['velocity']] for s in seed]

    # PAD/TRUNCATE to 16×3
    CONTEXT = 16
    if len(seq) < CONTEXT:
        padding = [[0.0,0.0,0.0]]*(CONTEXT - len(seq))
        seq_padded = padding + seq
    else:
        seq_padded = seq[-CONTEXT:]

    input_seq = np.array([seq_padded], dtype=np.float32)  # shape (1,16,3)

    melody, mdurs, mvels = [], [], []

    for _ in range(length):
        pred = melody_model.predict(input_seq, verbose=0)[0]
        pn, pd, pv = pred  # normalized outputs

        # DENORMALIZE
        note_int = int(round(note_scaler.inverse_transform([[pn]])[0][0]))
        dur_int  = max(1, int(round(duration_scaler.inverse_transform([[pd]])[0][0])))
        vel_int  = int(round(velocity_scaler.inverse_transform([[pv]])[0][0]))

        # SNAP to scale
        if is_semitone_in_scale(note_int, target_scale):
            final_note = note_int
        else:
            base    = note_int % 12
            sems    = [NOTE_TO_SEMITONE[n] for n in SCALE_DEGREES[target_scale]]
            nearest = min(sems, key=lambda s: abs(s-base))
            final_note = (note_int // 12)*12 + nearest

        melody.append(final_note)
        mdurs.append(dur_int)
        mvels.append(vel_int)

        # build next input row (normalized)
        new_scaled = np.hstack([
            note_scaler.transform([[final_note]]),
            duration_scaler.transform([[dur_int]]),
            velocity_scaler.transform([[vel_int]])
        ]).reshape(1,1,3)

        # slide window
        input_seq = np.concatenate([input_seq[:,1:,:], new_scaled], axis=1)

    return melody, mdurs, mvels

# ————————————————————————————————————————————————
from mido import MidiFile, MidiTrack, Message
import random
from typing import List

def create_midi_file(
    chords, chord_durs, chord_vels,
    melody, mel_durs, mel_vels,
    output_path='generated_music.mid'
):
    # — 1) Emparejar duraciones totales —
    total_chord   = sum(chord_durs)
    total_mel     = sum(mel_durs)
    scale_factor  = total_chord / total_mel if total_mel > 0 else 1.0

    # Ajustamos cada duración de la melodía
    mel_durs = [
        max(1, int(d * scale_factor))
        for d in mel_durs
    ]

    # — 2) Recopilar todos los eventos con tiempo absoluto —
    events = []
    abs_time = 0

    # Acordes (octava +1)
    for chord, dur, _ in zip(chords, chord_durs, chord_vels):
        for note in chord:
            # transponer acordes una octava
            note_up = clamp_to_range(note + 12)
            vel     = random.randint(63, 80)
            # note_on
            events.append((abs_time,
                           Message('note_on',  note=note_up, velocity=vel)))
            # note_off
            events.append((abs_time + dur,
                           Message('note_off', note=note_up, velocity=vel)))
        abs_time += dur

    # Melodía (sin cambio de octava)
    abs_time = 0
    for n, dur, _ in zip(melody, mel_durs, mel_vels):
        note_m = clamp_to_range(n)
        vel   = random.randint(53, 70)
        events.append((abs_time,
                       Message('note_on',  note=note_m, velocity=vel)))
        events.append((abs_time + dur,
                       Message('note_off', note=note_m, velocity=vel)))
        abs_time += dur

    # — 3) Ordenar e ir volcando a delta-times —
    events.sort(key=lambda x: x[0])
    mid   = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)

    last_time = 0
    for abs_t, msg in events:
        msg.time = abs_t - last_time
        track.append(msg)
        last_time = abs_t

    mid.save(output_path)
    print(f'✅ Música guardada en {output_path}')


# ————————————————————————————————————————————————
def generate_music(
    scale: str,
    output_path: str = 'generated_music.mid'
):
    target = scale.upper()
    if not target.endswith(('MINOR','MAJOR')):
        target = f"{target}MINOR"

    chords, cdurs, cvels = build_triad_chords(target)
    melody, mdurs, mvels = generate_melody(target)
    create_midi_file(chords, cdurs, cvels, melody, mdurs, mvels, output_path)
