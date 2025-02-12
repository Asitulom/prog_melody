import mido
import os
import json

def extract_midi_data(midi_path):
    """
    Extrae notas, tiempo de inicio, duración y velocidad (intensidad) de un archivo MIDI.
    """
    mid = mido.MidiFile(midi_path)
    notes = []

    for track in mid.tracks:
        current_time = 0  # Lleva el tiempo acumulado en ticks

        for msg in track:
            current_time += msg.time  # Suma el tiempo de cada mensaje MIDI

            if msg.type == 'note_on' and msg.velocity > 0:  # Detecta notas activadas
                notes.append({
                    "note": msg.note,  # Pitch (0-127)
                    "start_time": current_time,  # Inicio en ticks
                    "velocity": msg.velocity,  # Intensidad (0-127)
                    "duration": 0  # Se calculará después
                })

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Encuentra la nota previa y calcula la duración
                for note in reversed(notes):
                    if note["note"] == msg.note and note["duration"] == 0:
                        note["duration"] = current_time - note["start_time"]
                        break

    return notes


def process_midi_folder(folder_path, output_json="midi_data.json"):
    """
    Procesa todos los archivos MIDI en una carpeta y guarda los datos en un JSON.
    """
    midi_data = {}

    for file in os.listdir(folder_path):
        if file.endswith(".mid") or file.endswith(".midi"):
            midi_path = os.path.join(folder_path, file)
            midi_data[file] = extract_midi_data(midi_path)

    with open(output_json, "w") as f:
        json.dump(midi_data, f, indent=4)

    print(f"✅ Datos guardados en {output_json}")


# Ejemplo de uso
if __name__ == "__main__":
    midi_folder = r"C:\Users\Asier\Documents\PFG\prog_melody\ai\datasets\midi\sad"
    process_midi_folder(midi_folder, "sad_midi_data.json")
