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
    return melody_data  # Retorna la lista de notas [{"note": 60, "velocity": 100, "duration": 300}, ...]

# Convertir JSON a MIDI
def generate_midi(melody_data, output_midi_path="generated_melody.mid"):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    for note_info in melody_data:
        # Asegurar que los valores estén dentro del rango MIDI (0-127)
        midi_note = max(0, min(127, int(note_info["note"])))
        velocity = max(0, min(127, int(note_info.get("velocity", 100))))
        duration = max(0, min(1000, int(note_info.get("duration", 300))))  # Limitar la duración a un rango razonable

        # Crear eventos MIDI
        track.append(Message("note_on", note=midi_note, velocity=velocity, time=0))
        track.append(Message("note_off", note=midi_note, velocity=velocity, time=duration))

    midi.save(output_midi_path)
    print(f"✅ Archivo MIDI guardado en {output_midi_path}")

# ---------- EJECUCIÓN DEL SCRIPT ----------
if __name__ == "__main__":
    input_json = "ai/generated_melody.json"
    output_midi = "ai/generated_melody.mid"

    if not os.path.exists(input_json):
        print(f"❌ Error: El archivo {input_json} no existe. Genera la melodía primero.")
    else:
        melody_data = load_generated_melody(input_json)
        generate_midi(melody_data, output_midi)
