# backend/generate_music.py

import os
import json
import pickle
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from mido import MidiFile, MidiTrack, Message

# —————————————————————————————————————————————————————————————————————————
# Paths y configuración
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
AI_DIR               = os.path.join(BASE_DIR, '..', 'ai')

# Triads & progresiones (idéntico a antes)
LOWER_BOUND     = 24   # C3
UPPER_BOUND     = 84   # C6
TICKS_PER_BEAT  = 384

DURATION_PATTERNS = [
    [4,4,4,4],
    [2,6,2,6],
    [6,2,6,2],
]

PROGRESSIONS = [
    [0,3,4,0], [0,4,5,3], [0,6,5,6],
    [0,5,2,6], [0,2,3,4], [0,5,3,2],
    [0,2,6,3], [0,4,3,5], [0,5,4,3],
    [0,6,0,4],
]

# Cargar grados de escala
DEGREES_PATH   = os.path.join(AI_DIR, 'scale_degrees.json')
with open(DEGREES_PATH, 'r') as f:
    SCALE_DEGREES = json.load(f)

# Utilitarios de semitonos
NOTE_TO_SEMITONE = {
    "C":0, "C#":1, "Db":1, "D":2, "D#":3, "Eb":3,
    "E":4, "Fb":4, "F":5, "E#":5, "F#":6, "Gb":6,
    "G":7, "G#":8, "Ab":8, "A":9, "A#":10,"Bb":10,
    "B":11,"Cb":11
}

# —————————————————————————————————————————————————————————————————————————
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

# —————————————————————————————————————————————————————————————————————————
def load_pipeline(scale_type: str):
    """
    Carga el modelo LSTM y scalers para 'minor' o 'major',
    así como el JSON de datos correspondiente.
    """
    if scale_type == 'major':
        model_path   = os.path.join(AI_DIR, 'melody_model_happy.h5')
        note_scl_p   = os.path.join(AI_DIR, 'note_scaler_happy.pkl')
        dur_scl_p    = os.path.join(AI_DIR, 'duration_scaler_happy.pkl')
        vel_scl_p    = os.path.join(AI_DIR, 'velocity_scaler_happy.pkl')
        data_json    = os.path.join(AI_DIR, 'happy_midi_data.json')
    else:
        model_path   = os.path.join(AI_DIR, 'melody_model.h5')
        note_scl_p   = os.path.join(AI_DIR, 'note_scaler.pkl')
        dur_scl_p    = os.path.join(AI_DIR, 'duration_scaler.pkl')
        vel_scl_p    = os.path.join(AI_DIR, 'velocity_scaler.pkl')
        data_json    = os.path.join(AI_DIR, 'sad_midi_data.json')

    melody_model      = tf.keras.models.load_model(model_path)
    note_scaler       = pickle.load(open(note_scl_p, 'rb'))
    duration_scaler   = pickle.load(open(dur_scl_p,  'rb'))
    velocity_scaler   = pickle.load(open(vel_scl_p,  'rb'))
    return melody_model, note_scaler, duration_scaler, velocity_scaler, data_json

# —————————————————————————————————————————————————————————————————————————
def build_triad_chords(
    target_scale: str,
    length: int = 4
) -> Tuple[List[List[int]], List[int], List[int]]:
    prog       = random.choice(PROGRESSIONS)
    degrees    = [prog[i % len(prog)] for i in range(length)]
    pattern    = random.choice(DURATION_PATTERNS)
    chord_durs = [d * TICKS_PER_BEAT for d in pattern]

    ext_types = ['triad', '7th']
    weights   = [0.65, 0.35]
    ext_flags = random.choices(ext_types, weights=weights, k=length)

    notes = SCALE_DEGREES[target_scale]
    chords, vels = [], []

    for deg, ext in zip(degrees, ext_flags):
        root_name  = notes[deg]
        third_name = notes[(deg+2) % 7]
        fifth_name = notes[(deg+4) % 7]

        root_pc  = NOTE_TO_SEMITONE[root_name]
        third_pc = NOTE_TO_SEMITONE[third_name]
        fifth_pc = NOTE_TO_SEMITONE[fifth_name]

        root_3rd = clamp_to_range(root_pc + 36)
        root_4th = clamp_to_range(root_pc + 48)
        root_5th = clamp_to_range(root_pc + 60)

        third_5th = clamp_to_range(root_5th + ((third_pc - root_pc) % 12))
        fifth_5th = clamp_to_range(root_5th + ((fifth_pc - root_pc) % 12))

        chord_notes = [root_3rd, root_4th, root_5th, third_5th, fifth_5th]

        if ext == '7th':
            sev_name = notes[(deg+6) % 7]
            sev_pc   = NOTE_TO_SEMITONE[sev_name]
            sev_5th  = clamp_to_range(root_5th + ((sev_pc - root_pc) % 12))
            chord_notes.append(sev_5th)

        chords.append(chord_notes)
        vels.append(random.randint(63, 95))

    return chords, chord_durs, vels

# —————————————————————————————————————————————————————————————————————————
def generate_melody(
    target_scale: str,
    length: int = 8
) -> Tuple[List[int], List[int], List[int]]:
    """
    Genera una melodía LSTM ajustada al target_scale.
    Detecta major vs minor y carga el pipeline correspondiente.
    """
    # Determinar modo
    mode = 'major' if target_scale.endswith('MAJOR') else 'minor'
    model, note_scaler, duration_scaler, velocity_scaler, data_json = load_pipeline(mode)

    # Cargar datos preprocesados
    with open(data_json, 'r') as f:
        data = json.load(f)

    # Filtrar eventos del mismo modo
    candidates = [
        (scale, note_dict)
        for scale, content in data.items()
        for note_dict in content.get('melodias', [])
        if (mode == 'minor' and scale.endswith('MINOR'))
           or (mode == 'major' and scale.endswith('MAJOR'))
    ]
    random.shuffle(candidates)
    seed_notes = candidates[:3]

    # Semitone offset y preparar semilla
    seed = []
    for base_scale, m in seed_notes:
        offset = semitone_offset(base_scale, target_scale)
        seed.append({
            'note':     m['note'] + offset,
            'duration': m['duration'],
            'velocity': m['velocity']
        })

    # Crear secuencia de contexto (16 pasos)
    seq = [[s['note'], s['duration'], s['velocity']] for s in seed]
    CONTEXT = 16
    if len(seq) < CONTEXT:
        padding = [[0.0,0.0,0.0]] * (CONTEXT - len(seq))
        seq_padded = padding + seq
    else:
        seq_padded = seq[-CONTEXT:]

    input_seq = np.array([seq_padded], dtype=np.float32)  # (1,16,3)

    melody, mdurs, mvels = [], [], []
    for _ in range(length):
        pn, pd, pv = model.predict(input_seq, verbose=0)[0]
        note_int = int(round(note_scaler.inverse_transform([[pn]])[0][0]))
        dur_int  = max(1, int(round(duration_scaler.inverse_transform([[pd]])[0][0])))
        vel_int  = int(round(velocity_scaler.inverse_transform([[pv]])[0][0]))

        # Ajuste al semitono más cercano en la escala
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

        # Slide window
        new_scaled = np.hstack([
            note_scaler.transform([[final_note]]),
            duration_scaler.transform([[dur_int]]),
            velocity_scaler.transform([[vel_int]])
        ]).reshape(1,1,3)
        input_seq = np.concatenate([input_seq[:,1:,:], new_scaled], axis=1)

    return melody, mdurs, mvels

# —————————————————————————————————————————————————————————————————————————
def create_midi_file(
    chords, chord_durs, chord_vels,
    melody, mel_durs, mel_vels,
    output_path='generated_music.mid'
):
    # … (idéntico al original) …
    total_chord = sum(chord_durs)
    total_mel   = sum(mel_durs)
    scale_f     = total_chord / total_mel if total_mel > 0 else 1.0
    mel_durs    = [max(1, int(d * scale_f)) for d in mel_durs]

    events, abs_time = [], 0
    # Acordes
    for chord, dur, _ in zip(chords, chord_durs, chord_vels):
        for note in chord:
            note_up = clamp_to_range(note + 12)
            vel     = random.randint(63, 80)
            events.append((abs_time, Message('note_on',  note=note_up, velocity=vel)))
            events.append((abs_time + dur, Message('note_off', note=note_up, velocity=vel)))
        abs_time += dur

    # Melodía
    abs_time = 0
    for n, dur, _ in zip(melody, mel_durs, mel_vels):
        nm  = clamp_to_range(n)
        vel = random.randint(53, 70)
        events.append((abs_time,         Message('note_on',  note=nm, velocity=vel)))
        events.append((abs_time + dur,   Message('note_off', note=nm, velocity=vel)))
        abs_time += dur

    # Ordenar y convertir a delta-times
    events.sort(key=lambda x: x[0])
    mid   = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack(); mid.tracks.append(track)
    last_time = 0
    for abs_t, msg in events:
        msg.time = abs_t - last_time
        track.append(msg)
        last_time = abs_t

    mid.save(output_path)
    print(f'✅ Música guardada en {output_path}')

# —————————————————————————————————————————————————————————————————————————
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
