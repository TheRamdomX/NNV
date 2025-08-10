# NNV - Neural Network Visualizer

Herramienta web interactiva para visualizar, analizar y optimizar redes neuronales Keras.

## Características

- Carga de modelos Keras (`.keras`) y datos (`.npy`)
- Visualización gráfica de la arquitectura y conexiones
- Selección de capas y neuronas para análisis detallado
- Estadísticas de pesos y bias por neurona/capa
<!-- - Detección de hotspots y puntos muertos 
- Recomendaciones automáticas para poda y ampliación
- Interfaz moderna y responsiva (React + Tailwind) -->

## Instalación

### Backend (FastAPI + TensorFlow)

```bash
cd Back
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend (React + Vite)

```bash
cd Front
npm install
npm run dev
```

Accede a [http://localhost:5173](http://localhost:5173)

## Uso

1. **Carga tu modelo** `.keras` y tus datos `.npy` desde la interfaz.
2. **Visualiza la arquitectura** y selecciona capas/neuronas para ver detalles.
3. **Analiza pesos, bias y activaciones**.
<!-- 4. **Recibe recomendaciones** para optimizar la red (poda, ampliación).
5. **Guarda el modelo optimizado** si lo deseas. -->

## Estructura del proyecto

```
NNV/
├── Back/         # Backend FastAPI + TensorFlow
│   ├── app.py
│   ├── requirements.txt
├── Front/        # Frontend React + Vite
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/ui/
│   │   └── index.css
│   ├── package.json
└── README.md
```




