# NNV - Neural Network Visualizer

Aplicación fullstack para visualizar y analizar modelos de redes neuronales en formato `.keras`.

![React](https://img.shields.io/badge/React-18-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🎯 Características

- **📦 Inspección de archivos `.keras`**: Visualiza la estructura interna del archivo (config.json, metadata.json, model.weights.h5)
- **🧠 Detalles del modelo**: Información completa de capas, parámetros, optimizador y métricas
- **⚖️ Estadísticas de pesos**: Análisis detallado de tensores (min, max, media, std, shape)
- **🔗 Visualización interactiva**: Grafo de la red neuronal con neuronas clickeables
- **📍 Mini previsualización**: Vista rápida de la ubicación de capas seleccionadas

## 🖥️ Interfaz

La aplicación organiza la información en 4 pestañas:

| Tab | Descripción |
|-----|-------------|
| **Archivo** | Estructura del archivo .keras y metadatos |
| **Modelo** | Detalles de capas, configuración y mini previsualización |
| **Parámetros** | Estadísticas de pesos por capa y tensor |
| **Visualización** | Grafo interactivo de la red neuronal |

### Interacciones

- **Click en neurona**: Muestra/oculta conexiones y pesos
- **Hover en neurona**: Efecto de sombreado
- **Click en capa** (tab Modelo): Muestra detalles y ubicación en la red

## 🚀 Instalación

### Con Docker (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/TheRamdomX/NNV.git
cd NNV

# Iniciar los contenedores
docker-compose up -d

# Acceder a la aplicación
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
```

### Manual

#### Backend

```bash
cd Back
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd Front
npm install
npm run dev
```

## 📁 Estructura del Proyecto

```
NNV/
├── Back/
│   ├── app.py              # API FastAPI
│   ├── requirements.txt    # Dependencias Python
│   ├── Dockerfile
│   └── docker-compose.yml
├── Front/
│   ├── src/
│   │   ├── App.tsx         # Componente principal
│   │   ├── components/ui/  # Componentes UI
│   │   └── lib/utils.ts
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
├── docker-compose.yml      # Orquestación de servicios
└── README.md
```

## 🔧 API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/upload` | Carga un archivo .keras |
| POST | `/carga_parametros` | Carga parámetros desde archivo .npy |
| GET | `/model_info` | Obtiene información completa del modelo |

## 📊 Formato .keras

Los archivos `.keras` son archivos ZIP que contienen:

```
model.keras (ZIP)
├── config.json      # Arquitectura del modelo
├── metadata.json    # Versión de Keras, backend, fecha
└── model.weights.h5 # Pesos en formato HDF5
```

## 🛠️ Tecnologías

### Backend
- **FastAPI** - Framework web async
- **TensorFlow 2.19** - Carga y análisis de modelos
- **h5py** - Lectura de archivos HDF5
- **NumPy** - Operaciones numéricas

### Frontend
- **React 18** - UI declarativa
- **TypeScript** - Tipado estático
- **Vite** - Build tool
- **Tailwind CSS** - Estilos utility-first
- **Axios** - Cliente HTTP

## 📝 Uso

1. **Cargar modelo**: Selecciona un archivo `.keras` y haz clic en "Cargar modelo"
2. **Explorar tabs**: Navega entre Archivo, Modelo, Parámetros y Visualización
3. **Interactuar**: Haz clic en capas o neuronas para ver detalles
4. **(Opcional)** Cargar parámetros adicionales con archivo `.npy`

