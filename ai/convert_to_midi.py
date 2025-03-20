#convert_to_midi.py

import json
import mido
from mido import Message, MidiFile, MidiTrack
import os

# Mapeo de notas musicales a números MIDI
NOTE_MAPPING = {
    "C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64, "F": 65, "F#": 66,
    "G": 67, "G#": 68, "A": 69, "A#": 70, "B": 71
}

# Cargar la melodía generada desde JSON
def load_generated_melody(json_path):
    with open(json_path, "r") as file:
        melody_data = json.load(file)  # El archivo JSON contiene solo una lista de notas
    return melody_data  # Retorna la lista de notas ["C", "A", "D", "G", ...]

# Convertir JSON a MIDI
def generate_midi(melody_data, output_midi_path="generated_melody.mid"):
    midi = MidiFile()  # Crear archivo MIDI
    track = MidiTrack()  # Crear una pista
    midi.tracks.append(track)  # Agregar la pista al archivo

    # Parámetros de duración y velocidad predeterminados
    duration = 300  # Tiempo entre nota_on y nota_off (ajustable)
    velocity = 100  # Intensidad de la nota

    for note in melody_data:
        midi_note = NOTE_MAPPING.get(note, 60)  # Convertir nota a número MIDI (default: C=60)
        track.append(Message("note_on", note=midi_note, velocity=velocity, time=0))
        track.append(Message("note_off", note=midi_note, velocity=velocity, time=duration))

    # Guardar el archivo MIDI
    midi.save(output_midi_path)
    print(f"✅ Archivo MIDI guardado en {output_midi_path}")

# ---------- EJECUCIÓN DEL SCRIPT ----------
if __name__ == "__main__":
    input_json = r"C:\Users\Asier\Documents\PFG\prog_melody\ai\generated_melody.json"
    output_midi = r"C:\Users\Asier\Documents\PFG\prog_melody\ai\generated_melody.mid"

    if not os.path.exists(input_json):
        print(f"❌ Error: El archivo {input_json} no existe. Genera la melodía primero.")
    else:
        melody_data = load_generated_melody(input_json)
        generate_midi(melody_data, output_midi)