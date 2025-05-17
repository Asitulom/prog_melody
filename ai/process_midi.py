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

            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                if active_note and active_note[0] == msg.note:
                    # calculamos duración
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    midi_folder = os.path.join(BASE_DIR, "datasets", "midi", "sad")

    print(f"📂 Ruta del dataset: {midi_folder}")
    process_midi_folder(midi_folder, "sad_midi_data.json")
