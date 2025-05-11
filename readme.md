#README.MD

# Generador de Melodías con IA 🎵

Este proyecto es un generador básico de melodías utilizando un backend en **FastAPI** y un frontend en **HTML, CSS y JavaScript**. Permite a los usuarios generar melodías originales ajustando parámetros como el tempo, el tono y la emoción.


git clone https://github.com/Asitulom/prog_melody.git

## 🚀 Cómo ejecutar la aplicación


El usuario en el frontend solicita un midi con una escala concreta y una nota en concreto, despues el programa le devuelve un midi con acordes y melodias en la escala y nota que ha solicitado. Ejemplo: el usuario pide C y sad: recibe un midi de acordes y melodia en CMINOR



# 1. Ir a la carpeta del proyecto
cd C:\Users\Asier\Documents\PFG\prog_melody

# 2. Activar entorno virtual
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Procesar archivos MIDI (esto crea sad_midi_data.json)
cd ai
python process_midi.py

# 5. Entrenar los modelos LSTM
python train_melody_model.py
python train_chord_model.py

# 6.  
cd C:\Users\Asier\Documents\PFG\prog_melody\backend
python generate_music.py


# 7. Volver a la raíz del proyecto para lanzar el backend
cd ..
uvicorn backend.app:app --reload
http://127.0.0.1:8000/docs

# 8. En una nueva terminal (o nueva pestaña)
cd C:\Users\Asier\Documents\PFG\prog_melody\frontend
python -m http.server 8001

http://localhost:8001

 
# Extra. Para ver los usuarios de la base de datos
cd C:\Users\Asier\Documents\PFG\prog_melody
C:\sqlite\sqlite3.exe users.db
.tables      
SELECT * FROM users;

DELETE FROM valoraciones;



DOCKER:

cd C:\Users\Asier\Documents\PFG\prog_melody

docker-compose down

docker-compose up --build


**Backend**

http://127.0.0.1:8000/docs

**Frontend**

http://localhost:8001


Control + shift + r