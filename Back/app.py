from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uvicorn
import io
import tempfile
import os

app = FastAPI(
    title="Neural Network Visualizer API",
    description="API para cargar modelos Keras y datos, extraer activaciones, detectar 'dead'/'hotspots' y podar neuronas.",
)

# Almacenamiento global
model: tf.keras.Model | None = None
input_data: np.ndarray | None = None

# --- Model & Data Load Responses ---
class ModelLoadResponse(BaseModel):
    message: str

class DataLoadResponse(BaseModel):
    message: str
    shape: tuple[int, ...]

# --- Layer & Activation Models ---
class LayerInfo(BaseModel):
    index: int
    name: str
    output_shape: tuple[int, ...]
    activation: str | None = None

class ActivationRequest(BaseModel):
    layer_index: int
    threshold_low: float = 0.01
    threshold_high: float = 0.99

class NeuronStatus(BaseModel):
    neuron_index: int
    mean_activation: float
    status: str  # 'dead', 'normal', 'active'

# --- Pruning Models ---
class PruneConfig(BaseModel):
    layer_index: int
    threshold_low: float = 0.01

class PruneResponseItem(BaseModel):
    layer_index: int
    pruned_neurons: int
    total_neurons: int

# --- Endpoints ---
@app.post("/load_model", response_model=ModelLoadResponse)
async def load_model(file: UploadFile = File(...)):
    """Carga un modelo Keras (.keras o .h5)."""
    global model
    try:
        content = await file.read()
        print(f"[DEBUG /load_model]: Archivo recibido, tamaño: {len(content)} bytes")
        print(f"[DEBUG /load_model]: TensorFlow version: {tf.__version__}")
        
        # Guardar temporalmente el archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Cargar el modelo desde el archivo temporal
        model = tf.keras.models.load_model(tmp_file_path)
        
        # Limpiar el archivo temporal
        os.unlink(tmp_file_path)
        
        return ModelLoadResponse(message="Modelo cargado exitosamente.")
    except Exception as e:
        print(f"[ERROR /load_model]: {e}")
        raise HTTPException(status_code=400, detail=f"Error cargando modelo: {e}")

@app.post("/load_data", response_model=DataLoadResponse)
async def load_data(file: UploadFile = File(...)):
    """Carga datos de entrada en formato .npy."""
    global input_data
    try:
        content = await file.read()
        input_data = np.load(io.BytesIO(content), allow_pickle=True)
        return DataLoadResponse(message="Datos cargados exitosamente.", shape=input_data.shape)
    except Exception as e:
        print(f"[ERROR /load_data]: {e}")
        raise HTTPException(status_code=400, detail=f"Error cargando datos: {e}")

@app.get("/layers", response_model=list[LayerInfo])
async def list_layers():
    """Lista las capas del modelo con índice y forma de salida."""
    if model is None:
        print("[ERROR /layers]: Modelo no cargado.")
        raise HTTPException(status_code=400, detail="Modelo no cargado.")
    
    def safe_int(val):
        return int(val) if val is not None else -1

    layers_info = []
    for i, layer in enumerate(model.layers):
        try:
            # Para TensorFlow 2.19, obtener output_shape de manera diferente
            if hasattr(layer, 'output_shape'):
                output_shape = tuple(safe_int(x) for x in layer.output_shape)
            elif hasattr(layer, 'output'):
                output_shape = tuple(safe_int(x) for x in layer.output.shape)
            else:
                # Fallback: usar la forma de los pesos si están disponibles
                weights = layer.get_weights()
                if weights:
                    output_shape = (-1, weights[-1].shape[0])
                else:
                    output_shape = (-1,)
            
            activation = getattr(layer, 'activation', None)
            if activation is not None:
                if hasattr(activation, '__name__'):
                    activation = activation.__name__
                else:
                    activation = str(activation)
            layers_info.append(LayerInfo(
                index=i,
                name=layer.name,
                output_shape=output_shape,
                activation=activation
            ))
        except Exception as e:
            print(f"[WARNING /layers]: Error procesando capa {i} ({layer.name}): {e}")
            continue
    
    return layers_info

@app.post("/activations", response_model=list[NeuronStatus])
async def analyze_activations(req: ActivationRequest):
    """Analiza activaciones en una capa y clasifica neuronas."""
    if model is None or input_data is None:
        print("[ERROR /activations]: Modelo o datos no cargados.")
        raise HTTPException(status_code=400, detail="Modelo o datos no cargados.")
    try:
        layer = model.layers[req.layer_index]
        intermediate = tf.keras.Model(inputs=model.input, outputs=layer.output)
        activations = intermediate.predict(input_data)
        flat = activations.reshape(-1, activations.shape[-1])
        means = flat.mean(axis=0)
        statuses = []
        for i, m in enumerate(means):
            if m < req.threshold_low:
                status = 'dead'
            elif m > req.threshold_high:
                status = 'active'
            else:
                status = 'normal'
            statuses.append(NeuronStatus(neuron_index=i, mean_activation=float(m), status=status))
        return statuses
    except Exception as e:
        print(f"[ERROR /activations]: {e}")
        raise HTTPException(status_code=400, detail=f"Error analizando activaciones: {e}")

@app.post("/prune", response_model=list[PruneResponseItem])
async def prune_model(configs: list[PruneConfig]):
    """Poda neuronas cuyas activaciones promedio estén bajo el umbral indicado."""
    if model is None or input_data is None:
        print("[ERROR /prune]: Modelo o datos no cargados.")
        raise HTTPException(status_code=400, detail="Modelo o datos no cargados.")
    results: list[PruneResponseItem] = []
    try:
        for cfg in configs:
            layer = model.layers[cfg.layer_index]
            # Sólo capas Dense
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            # Obtener activaciones
            intermediate = tf.keras.Model(inputs=model.input, outputs=layer.output)
            acts = intermediate.predict(input_data)
            flat = acts.reshape(-1, acts.shape[-1])
            means = flat.mean(axis=0)
            # Identificar neuronas a podar
            to_prune = np.where(means < cfg.threshold_low)[0]
            weights, biases = layer.get_weights()
            # Poner a cero los pesos y bias de neuronas muertas
            weights[:, to_prune] = 0
            biases[to_prune] = 0
            layer.set_weights([weights, biases])
            results.append(PruneResponseItem(
                layer_index=cfg.layer_index,
                pruned_neurons=int(len(to_prune)),
                total_neurons=int(weights.shape[1])
            ))
        return results
    except Exception as e:
        print(f"[ERROR /prune]: {e}")
        raise HTTPException(status_code=400, detail=f"Error podando modelo: {e}")

@app.post("/save_model", response_model=ModelLoadResponse)
async def save_model(file_name: str):
    """Guarda el modelo actual en un archivo .keras local."""
    if model is None:
        print("[ERROR /save_model]: Modelo no cargado.")
        raise HTTPException(status_code=400, detail="Modelo no cargado.")
    try:
        path = f"{file_name}.keras"
        model.save(path)
        return ModelLoadResponse(message=f"Modelo guardado en {path}")
    except Exception as e:
        print(f"[ERROR /save_model]: {e}")
        raise HTTPException(status_code=400, detail=f"Error guardando modelo: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
