# Generador de Melodías con IA 🎵

Este proyecto es un generador básico de melodías utilizando un backend en **FastAPI** y un frontend en **HTML, CSS y JavaScript**. Permite a los usuarios generar melodías originales ajustando parámetros como el tempo, el tono y la emoción.

## 🚀 Cómo ejecutar la aplicación

1. **Clonar el repositorio:**

git clone https://github.com/Asitulom/prog_melody.git

2. **Instalar dependencias:**

cd C:\Users\Asier\Documents\PFG\prog_melody
pip install -r requirements.txt

3. **Ejecutar el backend:**

uvicorn backend.app:app --reload
http://127.0.0.1:8000/docs


4. **Ejecutar el frontend:**

cd C:\Users\Asier\Documents\PFG\prog_melody\frontend
python -m http.server 8001


5. **Abrir el frontend en el navegador:**

http://localhost:8001

6. **Leer archivos MIDI**

cd C:\Users\Asier\Documents\PFG\prog_melody\ai
python process_midi.py



##  Estructura del proyecto

```
PROG_MELODY/
    ├── ai/
    │   ├── generate.py        # Generación de melodías
    │   └── melody_model.py     # Modelo básico (aún no entrenado)
    ├── backend/
    │   ├── app.py             # Configuración del backend y rutas
    │   └── melody_service.py   # Lógica de generación de melodías
    ├── frontend/
    │   └── index.html        # Interfaz del usuario (HTML + JS)
    ├── readme.md              # Documentación del proyecto
    └── requirements.txt      # Dependencias del backend
```

## 🔧 Tecnologías utilizadas
- **Backend:** FastAPI (Python)
- **Frontend:** HTML, CSS, JavaScript (sin framework adicional)
- **Manejo de solicitudes HTTP:** Fetch API (en el frontend)

## 🛠️ Funcionalidades actuales
- Generar melodías dinámicas según los parámetros proporcionados (tempo, tono y emoción).
- Lógica básica para diferentes estados emocionales: `happy`, `sad` y `excited`.
- Visualización de la melodía generada en la interfaz gráfica.
