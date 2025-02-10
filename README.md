pip install -r requirements.txt

app.py (FastApi)

cd C:\Users\Asier\Documents\PFG\prog_melody
uvicorn backend.app:app --reload
http://127.0.0.1:8000/docs




cd C:\Users\Asier\Documents\PFG\prog_melody\frontend
python -m http.server 8001
http://localhost:8001


# Generador de Melodías con IA 🎵

Este proyecto es un generador básico de melodías utilizando un backend en **FastAPI** y un frontend en **HTML, CSS y JavaScript**. Permite a los usuarios generar melodías originales ajustando parámetros como el tempo, el tono y la emoción.

## 🚀 Cómo ejecutar la aplicación

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd prog_melody
   ```

2. **Instalar dependencias:**
   Ejecuta el siguiente comando para instalar las dependencias del backend especificadas en el archivo `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el backend:**
   Inicia el servidor FastAPI.
   ```bash
   uvicorn backend.app:app --reload
   ```

4. **Ejecutar el frontend:**
   Navega a la carpeta `frontend` y ejecuta el servidor local.
   ```bash
   cd frontend
   python -m http.server 8001
   ```

5. **Abrir el frontend en el navegador:**
   Abre tu navegador y visita **[http://localhost:8001](http://localhost:8001)**.

## 🔖 Estructura del proyecto

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

## ⚡️ Futuras mejoras (Fase 2 y Fase 3)
- Implementar un reproductor de audio (archivos MIDI o WAV).
- Almacenar melodías y feedback del usuario en una base de datos.
- Entrenar un modelo avanzado para la generación de melodías usando datasets reales.

## 📊 Feedback y contribuciones
Si tienes sugerencias o deseas contribuir, no dudes en abrir un **issue** o enviar un **pull request**.
