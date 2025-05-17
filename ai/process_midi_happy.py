# process_midi_happy.py

import mido
import os
import json
from collections import defaultdict

# Definición de escalas (solo mayores por ahora)
SCALES = {
    "AMAJOR":  ["A",  "B",  "C#", "D",  "E",  "F#", "G#"],
    "A#MAJOR": ["A#", "C",  "D",  "D#", "F",  "G",  "A"],
    "BMAJOR":  ["B",  "C#", "D#", "E",  "F#", "G#", "A#"],
    "CMAJOR":  ["C",  "D",  "E",  "F",  "G",  "A",  "B"],
    "C#MAJOR": ["C#", "D#", "F",  "F#", "G#", "A#", "C"],
    "DMAJOR":  ["D",  "E",  "F#", "G",  "A",  "B",  "C#"],
    "D#MAJOR": ["D#", "F",  "G",  "G#", "A#", "C",  "D"],
    "EMAJOR":  ["E",  "F#", "G#", "A",  "B",  "C#", "D#"],
    "FMAJOR":  ["F",  "G",  "A",  "A#", "C",  "D",  "E"],
    "F#MAJOR": ["F#", "G#", "A#", "B",  "C#", "D#", "F"],
    "GMAJOR":  ["G",  "A",  "B",  "C",  "D",  "E",  "F#"],
    "G#MAJOR": ["G#", "A#", "C",  "C#", "D#", "F",  "G"]
}

# Función para extraer escala del nombre del archivo
def get_scale_from_filename(filename):
    parts = filename.split(" - ")
    if len(parts) > 1:
        scale = parts[1].replace(".mid", "").strip()
        return scale.upper()
    return None

# Función para procesar un archivo MIDI (solo melodías)
def process_midi_file(midi_path):
    mid = mido.MidiFile(midi_path)
    melody = []
    current_time = 0
    active_note = None  # tuple (note, velocity, start_time)

    for track in mid.tracks:
        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                # iniciamos una nota
                active_note = (msg.note, msg.velocity, current_time)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if active_note and active_note[0] == msg.note:
                    note, velocity, start_time = active_note
                    duration = max(1, current_time - start_time)
                    melody.append({
                        "note": note,
                        "duration": duration,
                        "velocity": velocity
                    })
                active_note = None

    return melody

# Función para procesar toda la carpeta de melodías
def process_midi_folder(folder_path, output_json="midi_data.json"):
    data = defaultdict(lambda: {"melodias": []})
    subpath = os.path.join(folder_path, "melody")

    print(f"Buscando archivos en: {subpath}")
    if not os.path.exists(subpath):
        print(f"❌ No existe la carpeta: {subpath}")
        return

    for file in os.listdir(subpath):
        if file.endswith(".mid"):
            midi_path = os.path.join(subpath, file)
            print(f"Procesando archivo: {midi_path}")
            scale = get_scale_from_filename(file)
            if scale and scale in SCALES:
                print(f"Escala detectada: {scale} - Melodía")
                processed_data = process_midi_file(midi_path)
                data[scale]["melodias"].extend(processed_data)
            else:
                print(f"⚠️ Escala no reconocida o archivo mal nombrado: {file}")

    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Datos procesados y guardados en {output_json}")

if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    midi_folder = os.path.join(BASE_DIR, "datasets", "midi", "happy")

    print(f"📂 Ruta del dataset: {midi_folder}")
    process_midi_folder(midi_folder, "happy_midi_data.json")
