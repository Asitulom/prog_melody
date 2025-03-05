#convert_to_midi.py


import json
import mido
from mido import Message, MidiFile, MidiTrack

# Cargar la melodía generada desde JSON
def load_generated_melody(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)  # Cargar JSON como diccionario
    first_key = list(data.keys())[0]  # Obtener la primera clave (nombre del MIDI)
    return data[first_key]  # Devolver solo la lista de notas

# Convertir JSON a MIDI
def generate_midi(melody_data, output_midi_path="generated_melody.mid"):
    midi = MidiFile()  # Crear archivo MIDI
    track = MidiTrack()  # Crear una pista
    midi.tracks.append(track)  # Agregar la pista al archivo

    for note_info in melody_data:
        note = note_info["note"]
        velocity = note_info["velocity"]
        duration = note_info["duration"]
        
        track.append(Message("note_on", note=note, velocity=velocity, time=0))
        track.append(Message("note_off", note=note, velocity=velocity, time=duration))

    # Guardar el archivo MIDI
    midi.save(output_midi_path)
    print(f"✅ Archivo MIDI guardado en {output_midi_path}")

# ---------- EJECUCIÓN DEL SCRIPT ----------
if __name__ == "__main__":
    input_json = r"C:\Users\Asier\Documents\PFG\prog_melody\ai\sad_midi_data.json"
    output_midi = r"C:\Users\Asier\Documents\PFG\prog_melody\ai\generated_melody.mid"

    melody_data = load_generated_melody(input_json)
    generate_midi(melody_data, output_midi)
