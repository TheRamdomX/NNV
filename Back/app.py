# NNV Backend - API para análisis de modelos Keras
# Extrae información de archivos .keras (ZIP con config.json, metadata.json y weights.h5)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uvicorn
import logging
import io
import tempfile
import os
import zipfile
import json
import h5py
from typing import List, Optional, Dict, Any
from datetime import datetime

app = FastAPI()

# Logging básico para que logging.exception aparezca en los logs del contenedor
logging.basicConfig(level=logging.INFO)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global del modelo cargado
model: Optional[tf.keras.Model] = None
input_data: Optional[np.ndarray] = None
keras_file_info: Optional[Dict[str, Any]] = None

class ModelLoadResponse(BaseModel):
    message: str

class DataLoadResponse(BaseModel):
    message: str
    shape: tuple[int, ...]

# === Modelos Pydantic ===

class KerasFileStructure(BaseModel):
    """Estructura del archivo .keras"""
    files: List[str]
    config_json: Optional[Dict[str, Any]] = None
    metadata_json: Optional[Dict[str, Any]] = None

class WeightTensorInfo(BaseModel):
    """Información de un tensor de pesos"""
    name: str
    shape: List[int]
    dtype: str
    size: int
    min_value: float
    max_value: float
    mean_value: float
    std_value: float

class LayerWeightsInfo(BaseModel):
    """Pesos agrupados por capa"""
    layer_name: str
    tensors: List[WeightTensorInfo]

class LayerRawTensor(BaseModel):
    """Tensor individual con estadísticas"""
    name: str
    shape: List[int]
    size: int
    dtype: str
    stats: Dict[str, float]
    
class LayerFullWeights(BaseModel):
    """Información completa de pesos de una capa"""
    layer_index: int
    layer_name: str
    layer_type: str
    trainable: bool
    tensors: List[LayerRawTensor]
    total_params: int
    memory_mb: float

class OptimizerStateInfo(BaseModel):
    """Estado del optimizador"""
    type: str
    config: Dict[str, Any]
    variables: Optional[List[str]] = None

class LayerInfo(BaseModel):
    index: int
    name: str
    output_shape: list[int]
    activation: str | None = None
    type: str | None = None
    config: dict | None = None
    neuron_count: int | None = None
    input_shape: Optional[list[int]] = None
    trainable: Optional[bool] = None
    dtype: Optional[str] = None

class OptimizerInfo(BaseModel):
    type: str
    config: dict

class ModelMetadata(BaseModel):
    model_name: str
    created_at: str
    trainable_params: int
    non_trainable_params: int
    total_params: int
    keras_version: Optional[str] = None
    backend: Optional[str] = None
    date_saved: Optional[str] = None

class ModelInfo(BaseModel):
    summary: str
    layers: List[LayerInfo]
    optimizer: OptimizerInfo
    loss: str
    metrics: List[str]
    metadata: ModelMetadata
    layer_weights: List[LayerFullWeights]
    # Pesos y biases pueden tener formas variadas (vectores, matrices, tensores).
    # Usamos tipos flexibles para evitar errores de validación cuando una capa
    # tiene tensores escalar/1D/2D/ND.
    weights: List[Any]
    biases: List[Any]
    keras_file_structure: Optional[KerasFileStructure] = None
    detailed_weights: Optional[List[LayerWeightsInfo]] = None


# === Funciones de extracción ===

def extract_keras_zip_info(keras_path: str) -> Dict[str, Any]:
    """Extrae estructura y metadatos del archivo .keras (ZIP)"""
    info = {
        "files": [],
        "config_json": None,
        "metadata_json": None,
        "weights_h5_path": None
    }
    
    try:
        with zipfile.ZipFile(keras_path, 'r') as zf:
            info["files"] = zf.namelist()
            
            # Leer config.json
            if 'config.json' in info["files"]:
                with zf.open('config.json') as f:
                    info["config_json"] = json.load(f)
            
            # Leer metadata.json (puede tener diferentes nombres)
            metadata_files = [f for f in info["files"] if 'metadata' in f.lower() and f.endswith('.json')]
            if metadata_files:
                with zf.open(metadata_files[0]) as f:
                    info["metadata_json"] = json.load(f)
            
            # Identificar archivo de pesos
            weights_files = [f for f in info["files"] if f.endswith('.h5')]
            if weights_files:
                info["weights_h5_path"] = weights_files[0]
                
    except zipfile.BadZipFile:
        pass  # El archivo no es un ZIP válido
    except Exception:
        pass  # Error al procesar el archivo
    
    return info


def extract_weights_info_from_h5(keras_path: str, weights_filename: str) -> List[Dict[str, Any]]:
    """Extrae estadísticas de pesos desde el archivo H5."""
    weights_info = []
    
    try:
        with zipfile.ZipFile(keras_path, 'r') as zf:
            with zf.open(weights_filename) as h5_file:
                h5_content = io.BytesIO(h5_file.read())
                
                with h5py.File(h5_content, 'r') as hf:
                    def visit_weights(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            data = obj[:]
                            weights_info.append({
                                "name": name,
                                "shape": list(obj.shape),
                                "dtype": str(obj.dtype),
                                "size": obj.size,
                                "min_value": float(np.min(data)) if data.size > 0 else 0,
                                "max_value": float(np.max(data)) if data.size > 0 else 0,
                                "mean_value": float(np.mean(data)) if data.size > 0 else 0,
                                "std_value": float(np.std(data)) if data.size > 0 else 0
                            })
                    
                    hf.visititems(visit_weights)
                    
    except Exception:
        pass  # Error al leer pesos H5
    
    return weights_info


def parse_layer_config(config_json: Dict) -> List[Dict[str, Any]]:
    """Parsea la configuración de capas desde config.json"""
    layers_info = []
    
    if not config_json:
        return layers_info
    
    model_config = config_json.get("config", config_json)
    
    if "layers" in model_config:
        for i, layer in enumerate(model_config["layers"]):
            layer_config = layer.get("config", {})
            layer_class = layer.get("class_name", "Unknown")
            
            info = {
                "index": i,
                "class_name": layer_class,
                "name": layer_config.get("name", f"layer_{i}"),
                "config": layer_config,
                "build_config": layer.get("build_config", {}),
                "inbound_nodes": layer.get("inbound_nodes", [])
            }
            
            # Extraer información específica según el tipo de capa
            if layer_class == "Dense":
                info["units"] = layer_config.get("units")
                info["activation"] = layer_config.get("activation")
                info["use_bias"] = layer_config.get("use_bias", True)
            elif layer_class == "InputLayer":
                info["batch_input_shape"] = layer_config.get("batch_input_shape")
                info["dtype"] = layer_config.get("dtype")
            elif layer_class == "Dropout":
                info["rate"] = layer_config.get("rate")
            elif layer_class in ["Conv2D", "Conv1D"]:
                info["filters"] = layer_config.get("filters")
                info["kernel_size"] = layer_config.get("kernel_size")
                info["strides"] = layer_config.get("strides")
                info["padding"] = layer_config.get("padding")
                info["activation"] = layer_config.get("activation")
            elif layer_class in ["MaxPooling2D", "AveragePooling2D"]:
                info["pool_size"] = layer_config.get("pool_size")
                info["strides"] = layer_config.get("strides")
            elif layer_class == "Flatten":
                pass  # Flatten no tiene parámetros especiales
            elif layer_class in ["BatchNormalization"]:
                info["axis"] = layer_config.get("axis")
                info["momentum"] = layer_config.get("momentum")
                info["epsilon"] = layer_config.get("epsilon")
            
            layers_info.append(info)
    
    return layers_info

@app.post("/load_model", response_model=ModelLoadResponse)
async def load_model(file: UploadFile = File(...)):
    """Carga un modelo .keras y extrae su información"""
    global model, keras_file_info
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Extraer información del archivo ZIP
        keras_file_info = extract_keras_zip_info(tmp_path)
        
        if keras_file_info["config_json"]:
            parsed_layers = parse_layer_config(keras_file_info["config_json"])
            keras_file_info["parsed_layers"] = parsed_layers
        
        if keras_file_info["weights_h5_path"]:
            weights_info = extract_weights_info_from_h5(tmp_path, keras_file_info["weights_h5_path"])
            keras_file_info["weights_info"] = weights_info
        
        # Cargar modelo con TensorFlow
        model = tf.keras.models.load_model(tmp_path)
        os.unlink(tmp_path)
        
        return ModelLoadResponse(message="Modelo cargado exitosamente")
    except Exception as e:
        logging.exception("Error cargando modelo")
        raise HTTPException(status_code=400, detail=f"Error cargando modelo: {e}")

@app.get("/model_info", response_model=ModelInfo)
async def model_info():
    """Retorna información completa del modelo cargado"""
    global keras_file_info
    if model is None:
        raise HTTPException(status_code=400, detail="Modelo no cargado")
    try:
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        summary_text = "\n".join(summary_list)

        layers = []
        weights = []
        biases = []
        layer_weights_list = []

        for i, layer in enumerate(model.layers):
            shape = getattr(layer, 'output_shape', [-1])
            if isinstance(shape, tuple):
                shape = list(shape)
            elif hasattr(shape, 'as_list'):
                shape = shape.as_list()
            else:
                shape = [-1]

            # Input shape
            input_shape = getattr(layer, 'input_shape', None)
            if isinstance(input_shape, tuple):
                input_shape = list(input_shape)
            elif hasattr(input_shape, 'as_list'):
                input_shape = input_shape.as_list()

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
                neuron_count=neuron_count,
                input_shape=input_shape,
                trainable=layer.trainable,
                dtype=str(layer.dtype) if hasattr(layer, 'dtype') else None
            ))

            # Extraer tensores de la capa
            try:
                w = layer.get_weights()
                layer_tensors = []
                total_layer_params = 0
                weight_names = [v.name for v in layer.weights] if hasattr(layer, 'weights') else []
                
                for idx, tensor in enumerate(w):
                    tensor_array = np.array(tensor)
                    tensor_size = tensor_array.size
                    total_layer_params += tensor_size
                    
                    if idx < len(weight_names):
                        tensor_name = weight_names[idx].split('/')[-1].replace(':0', '')
                    else:
                        tensor_name = f"tensor_{idx}"
                    
                    sparsity = float(np.sum(tensor_array == 0) / tensor_size) if tensor_size > 0 else 0
                    
                    layer_tensors.append(LayerRawTensor(
                        name=tensor_name,
                        shape=list(tensor_array.shape),
                        size=tensor_size,
                        dtype=str(tensor_array.dtype),
                        stats={
                            "min": float(np.min(tensor_array)) if tensor_size > 0 else 0,
                            "max": float(np.max(tensor_array)) if tensor_size > 0 else 0,
                            "mean": float(np.mean(tensor_array)) if tensor_size > 0 else 0,
                            "std": float(np.std(tensor_array)) if tensor_size > 0 else 0,
                            "sparsity": sparsity
                        }
                    ))
                
                memory_mb = (total_layer_params * 4) / (1024 * 1024)
                
                layer_weights_list.append(LayerFullWeights(
                    layer_index=i,
                    layer_name=layer.name,
                    layer_type=layer_type,
                    trainable=layer.trainable,
                    tensors=layer_tensors,
                    total_params=total_layer_params,
                    memory_mb=round(memory_mb, 4)
                ))
                
                # Compatibilidad con estructura anterior
                if len(w) >= 1:
                    weights.append(w[0].tolist())
                else:
                    weights.append([])
                if len(w) >= 2:
                    biases.append(w[1].tolist())
                else:
                    biases.append([])
                    
            except Exception:
                weights.append([])
                biases.append([])
                layer_weights_list.append(LayerFullWeights(
                    layer_index=i,
                    layer_name=layer.name,
                    layer_type=layer_type,
                    trainable=layer.trainable,
                    tensors=[],
                    total_params=0,
                    memory_mb=0
                ))

        opt = model.optimizer
        optimizer_info = OptimizerInfo(
            type=opt.__class__.__name__,
            config=opt.get_config()
        ) if opt else OptimizerInfo(type="Desconocido", config={})

        trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
        non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_variables]))

        # Extraer metadatos del archivo .keras
        keras_version = None
        backend = None
        date_saved = None
        
        if keras_file_info and keras_file_info.get("metadata_json"):
            meta = keras_file_info["metadata_json"]
            keras_version = meta.get("keras_version")
            backend = meta.get("backend")
            date_saved = meta.get("date_saved")

        metadata = ModelMetadata(
            model_name=model.name,
            created_at=datetime.now().isoformat(),
            trainable_params=trainable,
            non_trainable_params=non_trainable,
            total_params=model.count_params(),
            keras_version=keras_version,
            backend=backend,
            date_saved=date_saved
        )

        # Extraer métricas
        metric_names = []
        if model.metrics:
            for m in model.metrics:
                if hasattr(m, 'name'):
                    metric_names.append(m.name)
                elif isinstance(m, str):
                    metric_names.append(m)
                elif hasattr(m, '__name__'):
                    metric_names.append(m.__name__)
                else:
                    metric_names.append(str(m))

        # Estructura del archivo .keras
        keras_structure = None
        if keras_file_info:
            keras_structure = KerasFileStructure(
                files=keras_file_info.get("files", []),
                config_json=keras_file_info.get("config_json"),
                metadata_json=keras_file_info.get("metadata_json")
            )

        # Información detallada de pesos
        detailed_weights = None
        if keras_file_info and keras_file_info.get("weights_info"):
            weights_by_layer = {}
            for w_info in keras_file_info["weights_info"]:
                parts = w_info["name"].split("/")
                layer_name = parts[0] if len(parts) > 1 else "unknown"
                if layer_name not in weights_by_layer:
                    weights_by_layer[layer_name] = []
                weights_by_layer[layer_name].append(WeightTensorInfo(
                    name=w_info["name"],
                    shape=w_info["shape"],
                    dtype=w_info["dtype"],
                    size=w_info["size"],
                    min_value=w_info["min_value"],
                    max_value=w_info["max_value"],
                    mean_value=w_info["mean_value"],
                    std_value=w_info["std_value"]
                ))
            
            detailed_weights = [
                LayerWeightsInfo(layer_name=name, tensors=tensors)
                for name, tensors in weights_by_layer.items()
            ]

        return ModelInfo(
            summary=summary_text,
            layers=layers,
            optimizer=optimizer_info,
            loss=model.loss if hasattr(model, 'loss') else "N/A",
            metrics=metric_names,
            metadata=metadata,
            layer_weights=layer_weights_list,
            weights=weights,
            biases=biases,
            keras_file_structure=keras_structure,
            detailed_weights=detailed_weights
        )
    except Exception as e:
        logging.exception("Error extrayendo información del modelo")
        raise HTTPException(status_code=500, detail=f"Error extrayendo información del modelo: {e}")


@app.get("/keras_file_structure")
async def get_keras_file_structure():
    """Retorna la estructura interna del archivo .keras (contenido del ZIP)."""
    global keras_file_info
    if keras_file_info is None:
        raise HTTPException(status_code=400, detail="No hay archivo .keras cargado")
    
    return {
        "files": keras_file_info.get("files", []),
        "has_config": keras_file_info.get("config_json") is not None,
        "has_metadata": keras_file_info.get("metadata_json") is not None,
        "has_weights": keras_file_info.get("weights_h5_path") is not None,
        "weights_file": keras_file_info.get("weights_h5_path")
    }


@app.get("/config_json")
async def get_config_json():
    """Retorna el contenido de config.json del archivo .keras."""
    global keras_file_info
    if keras_file_info is None or keras_file_info.get("config_json") is None:
        raise HTTPException(status_code=400, detail="No hay config.json disponible")
    
    return keras_file_info["config_json"]


@app.get("/metadata_json")
async def get_metadata_json():
    """Retorna el contenido de metadata.json del archivo .keras."""
    global keras_file_info
    if keras_file_info is None or keras_file_info.get("metadata_json") is None:
        raise HTTPException(status_code=400, detail="No hay metadata.json disponible")
    
    return keras_file_info["metadata_json"]


@app.get("/weights_structure")
async def get_weights_structure():
    """Retorna información detallada sobre la estructura de pesos del modelo."""
    global keras_file_info
    if keras_file_info is None or keras_file_info.get("weights_info") is None:
        raise HTTPException(status_code=400, detail="No hay información de pesos disponible")
    
    weights_info = keras_file_info["weights_info"]
    total_params = sum(w["size"] for w in weights_info)
    
    return {
        "total_tensors": len(weights_info),
        "total_parameters": total_params,
        "tensors": weights_info
    }


@app.get("/layer/{layer_index}/weights")
async def get_layer_weights(layer_index: int):
    """Retorna los pesos completos de una capa específica."""
    if model is None:
        raise HTTPException(status_code=400, detail="Modelo no cargado")
    
    if layer_index < 0 or layer_index >= len(model.layers):
        raise HTTPException(status_code=400, detail=f"Índice de capa inválido: {layer_index}")
    
    layer = model.layers[layer_index]
    weights = layer.get_weights()
    
    result = {
        "layer_name": layer.name,
        "layer_type": layer.__class__.__name__,
        "weights": []
    }
    
    for i, w in enumerate(weights):
        w_array = np.array(w)
        result["weights"].append({
            "index": i,
            "shape": list(w_array.shape),
            "dtype": str(w_array.dtype),
            "size": w_array.size,
            "data": w_array.tolist(),
            "stats": {
                "min": float(np.min(w_array)),
                "max": float(np.max(w_array)),
                "mean": float(np.mean(w_array)),
                "std": float(np.std(w_array))
            }
        })
    
    return result


@app.get("/optimizer_state")
async def get_optimizer_state():
    """Retorna información detallada del estado del optimizador."""
    if model is None:
        raise HTTPException(status_code=400, detail="Modelo no cargado")
    
    opt = model.optimizer
    if opt is None:
        raise HTTPException(status_code=400, detail="El modelo no tiene optimizador")
    
    result = {
        "type": opt.__class__.__name__,
        "config": opt.get_config(),
        "variables": []
    }
    
    if hasattr(opt, 'variables'):
        for var in opt.variables:
            result["variables"].append({
                "name": var.name,
                "shape": list(var.shape),
                "dtype": str(var.dtype)
            })
    
    return result


@app.post("/carga_parametros")
async def carga_parametros(npy: UploadFile = File(...)):
    """Carga parámetros adicionales desde un archivo .npy."""
    global input_data
    try:
        content = await npy.read()
        npy_buffer = io.BytesIO(content)
        input_data = np.load(npy_buffer, allow_pickle=False)
        return DataLoadResponse(
            message="Parámetros cargados exitosamente",
            shape=tuple(input_data.shape)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error cargando parámetros: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
