#generate.py

import random

def generate_melody_from_parameters(tempo, tone, emotion):
    # Notas posibles basadas en la escala de Do mayor
    notes = ["C", "D", "E", "F", "G", "A", "B"]

    # Validar y ajustar el tempo mínimo
    if tempo <= 0:
        tempo = 60  # Valor mínimo por defecto si el usuario introduce 0 o negativo

    # Ajustar la longitud de la melodía
    melody_length = max(5, 5 + int(tempo / 40))  # Siempre al menos 5 notas

    # Generar la melodía
    melody = [random.choice(notes) for _ in range(melody_length)]

    # Lógica según la emoción
    if emotion.lower() == "happy":
        melody += ["C", "E", "G"]
    elif emotion.lower() == "sad":
        melody += ["A", "D", "F"]
    elif emotion.lower() == "excited":
        melody += [random.choice(["C", "G", "B"])] * 2
    
    return melody
