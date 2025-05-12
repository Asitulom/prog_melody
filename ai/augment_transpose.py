# ai/augment_transpose.py
import json, os

# Cargamos el JSON que generaste con process_midi.py
IN_PATH  = os.path.join(os.path.dirname(__file__), "sad_midi_data.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "sad_midi_data_augmented.json")

with open(IN_PATH, "r") as f:
    raw = json.load(f)

augmented = {"AMINOR": {"melodias": [], "acordes": []}}
# (puedes agrupar por escala o aplastar todo en un único array de melodías)

# Para cada escala y cada melodía, generamos 12 transposiciones
for scale, block in raw.items():
    for note in block["melodias"]:
        orig = note["note"]
        for shift in range(12):
            augmented.setdefault(scale, {"melodias": [], "acordes": []})
            augmented[scale]["melodias"].append({
                "note":    (orig + shift) % 128,
                "duration": note["duration"],
                "velocity": note["velocity"]
            })

# Volcamos el JSON aumentado
with open(OUT_PATH, "w") as f:
    json.dump(augmented, f, indent=2)

print(f"✅ Augmented data saved to {OUT_PATH}")
