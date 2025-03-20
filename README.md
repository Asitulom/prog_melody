#README.MD

# Generador de Melodías con IA 🎵

Este proyecto es un generador básico de melodías utilizando un backend en **FastAPI** y un frontend en **HTML, CSS y JavaScript**. Permite a los usuarios generar melodías originales ajustando parámetros como el tempo, el tono y la emoción.


## 🚀 Cómo ejecutar la aplicación


cd C:\Users\Asier\Documents\PFG\prog_melody
## python -m venv venv
venv\Scripts\activate


1. **Clonar el repositorio:**

git clone https://github.com/Asitulom/prog_melody.git

2. **Instalar dependencias:**

cd C:\Users\Asier\Documents\PFG\prog_melody
pip install -r requirements.txt
pip install mido

3. **Leer archivos MIDI**

cd C:\Users\Asier\Documents\PFG\prog_melody\ai
python process_midi.py
✅ Datos guardados en sad_midi_data.json

4. **Entrenar modelo**

python melody_model.py
✅ Modelo entrenado y guardado como melody_model.h5

4. **Generar melodía**

python generate.py

5. **JSON a MIDI**

python convert_to_midi.py
✔ Esto debe generar un archivo generated_melody.mid

6. **Iniciar el backend**

cd C:\Users\Asier\Documents\PFG\prog_melody
uvicorn backend.app:app --reload
http://127.0.0.1:8000/docs

6. **Iniciar el frontend**

cd C:\Users\Asier\Documents\PFG\prog_melody\frontend
python -m http.server 8001
http://localhost:8001




DOCKER:

cd C:\Users\Asier\Documents\PFG\prog_melody

docker-compose down

docker-compose up --build


**Backend**

http://127.0.0.1:8000/docs

**Frontend**

 http://localhost:8001
