#process_midi.py

import mido
import os
import json
from collections import defaultdict

# Definición de escalas (solo menores por ahora)
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
# Mapeo de notas MIDI a nombres
MIDI_TO_NOTE = {i: v for i, v in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] * 11)}

# Inverso del diccionario
NOTE_TO_MIDI = {v: k for k, v in MIDI_TO_NOTE.items()}

# Margen para considerar notas como parte del mismo acorde (en ticks)
CHORD_THRESHOLD = 30

# Función para extraer escala del nombre del archivo
def get_scale_from_filename(filename):
    parts = filename.split("-")
    if len(parts) > 1:
        scale = parts[1].replace(".mid", "").strip()
        return scale.upper()
    return None

# Función para procesar un archivo MIDI
def process_midi_file(midi_path):
    mid = mido.MidiFile(midi_path)
    chords = []
    melody = []
    current_time = 0
    active_notes = []

    for track in mid.tracks:
        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes.append((msg.note, msg.velocity, current_time))

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if len(active_notes) >= 3:
                    start_times = [n[2] for n in active_notes]
                    min_time = min(start_times)
                    max_time = max(start_times)

                    if max_time - min_time <= CHORD_THRESHOLD:
                        chords.append({"notes": [n[0] for n in active_notes], "duration": max_time - min_time, "velocity": max([n[1] for n in active_notes])})
                    else:
                        for note in active_notes:
                            melody.append({"note": note[0], "duration": current_time - note[2], "velocity": note[1]})
                elif len(active_notes) == 1:
                    note = active_notes[0]
                    melody.append({"note": note[0], "duration": current_time - note[2], "velocity": note[1]})

                active_notes = []

    return chords, melody

# Función para procesar toda la carpeta
def process_midi_folder(folder_path, output_json="midi_data.json"):
    data = defaultdict(lambda: {"acordes": [], "melodia": []})

    for file in os.listdir(folder_path):
        if file.endswith(".mid"):
            midi_path = os.path.join(folder_path, file)
            scale = get_scale_from_filename(file)
            if scale and scale in SCALES:
                print(f"Procesando: {midi_path} - Escala: {scale}")
                chords, melody = process_midi_file(midi_path)
                data[scale]["acordes"].extend(chords)
                data[scale]["melodia"].extend(melody)

    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Datos procesados y guardados en {output_json}")

# Ejecución del script
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    midi_folder = os.path.join(BASE_DIR, "datasets", "midi", "sad")

    # Mostrar la ruta a procesar
    print(f"Ruta a procesar: {midi_folder}")

    process_midi_folder(midi_folder, "sad_midi_data.json")
