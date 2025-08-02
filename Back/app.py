# Backend: main.py (FastAPI + TensorFlow + visualización de red neuronal con pesos y bias)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uvicorn
import io
import tempfile
import os
from typing import List, Optional
from datetime import datetime

app = FastAPI()

# Habilitar CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[tf.keras.Model] = None
input_data: Optional[np.ndarray] = None

class ModelLoadResponse(BaseModel):
    message: str

class DataLoadResponse(BaseModel):
    message: str
    shape: tuple[int, ...]

class LayerInfo(BaseModel):
    index: int
    name: str
    output_shape: list[int]
    activation: str | None = None
    type: str | None = None
    config: dict | None = None
    neuron_count: int | None = None  # <-- Nuevo campo

class OptimizerInfo(BaseModel):
    type: str
    config: dict

class ModelMetadata(BaseModel):
    model_name: str
    created_at: str
    trainable_params: int
    non_trainable_params: int
    total_params: int

class ModelInfo(BaseModel):
    summary: str
    layers: List[LayerInfo]
    optimizer: OptimizerInfo
    loss: str
    metrics: List[str]
    metadata: ModelMetadata
    weights: List[List[List[float]]]
    biases: List[List[float]]

@app.post("/load_model", response_model=ModelLoadResponse)
async def load_model(file: UploadFile = File(...)):
    global model
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        model = tf.keras.models.load_model(tmp_path)
        os.unlink(tmp_path)
        return ModelLoadResponse(message="Modelo cargado exitosamente")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error cargando modelo: {e}")

@app.get("/model_info", response_model=ModelInfo)
async def model_info():
    if model is None:
        print("Modelo no cargado")
        raise HTTPException(status_code=400, detail="Modelo no cargado")
    try:
        print("Obteniendo summary del modelo...")
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        summary_text = "\n".join(summary_list)

        layers = []
        weights = []
        biases = []

        print("Iterando sobre las capas del modelo...")
        for i, layer in enumerate(model.layers):
            print(f"Procesando capa {i}: {layer.name}")
            shape = getattr(layer, 'output_shape', [-1])
            if isinstance(shape, tuple):
                shape = list(shape)
            elif hasattr(shape, 'as_list'):
                shape = shape.as_list()
            else:
                shape = [-1]

            activation = getattr(layer, 'activation', None)
            act_name = activation.__name__ if callable(activation) else str(activation)

            # Determina la cantidad de neuronas
            neuron_count = None
            layer_type = layer.__class__.__name__
            config = layer.get_config() if hasattr(layer, 'get_config') else None
            if layer_type == "Dense" and config and "units" in config:
                neuron_count = config["units"]
            elif layer_type == "InputLayer" and config and "batch_input_shape" in config:
                shape_conf = config["batch_input_shape"]
                if isinstance(shape_conf, (list, tuple)) and len(shape_conf) > 1 and isinstance(shape_conf[1], int):
                    neuron_count = shape_conf[1]
            elif isinstance(shape, (list, tuple)) and len(shape) > 0 and isinstance(shape[-1], int):
                neuron_count = shape[-1]

            layers.append(LayerInfo(
                index=i,
                name=layer.name,
                output_shape=shape,
                activation=act_name,
                type=layer_type,
                config=config,
                neuron_count=neuron_count
            ))

            try:
                w = layer.get_weights()
                if len(w) == 2:
                    weights.append(w[0].tolist())
                    biases.append(w[1].tolist())
                else:
                    weights.append([])
                    biases.append([])
            except Exception as e:
                print(f"Error obteniendo pesos/bias de la capa {layer.name}: {e}")
                weights.append([])
                biases.append([])

        print("Obteniendo información del optimizador...")
        opt = model.optimizer
        optimizer_info = OptimizerInfo(
            type=opt.__class__.__name__,
            config=opt.get_config()
        ) if opt else OptimizerInfo(type="Desconocido", config={})

        print("Calculando parámetros del modelo...")
        trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
        non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_variables]))

        metadata = ModelMetadata(
            model_name=model.name,
            created_at=datetime.now().isoformat(),
            trainable_params=trainable,
            non_trainable_params=non_trainable,
            total_params=model.count_params()
        )

        print("Preparando respuesta final...")
        # Depuración de métricas
        metric_names = []
        if model.metrics:
            for m in model.metrics:
                print(f"Procesando métrica: {m} ({type(m)})")
                # Solo incluir métricas con nombre legible
                if hasattr(m, 'name'):
                    metric_names.append(m.name)
                elif isinstance(m, str):
                    metric_names.append(m)
                elif hasattr(m, '__name__'):
                    metric_names.append(m.__name__)
                else:
                    metric_names.append(str(m))
        else:
            metric_names = []

        return ModelInfo(
            summary=summary_text,
            layers=layers,
            optimizer=optimizer_info,
            loss=model.loss if hasattr(model, 'loss') else "N/A",
            metrics=metric_names,
            metadata=metadata,
            weights=weights,
            biases=biases
        )
    except Exception as e:
        print(f"Error extrayendo información del modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error extrayendo información del modelo: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
