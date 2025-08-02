// Frontend: pages/index.tsx (React + estructuras personalizadas con tooltip + datos del modelo + selección de neuronas + neuronas de entrada/salida visibles + info extendida + pesos + conexión a backend)

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import axios from 'axios';

// Axios configurado con el backend en 0.0.0.0:8000
const api = axios.create({
  baseURL: 'http://localhost:8000',
});

interface Layer {
  index: number;
  name: string;
  output_shape?: number[];
  activation?: string | null;
  type?: string;
  config?: any;
  neuron_count?: number;
}

interface Neuron {
  id: string;
  x: number;
  y: number;
  layerIndex: number;
  neuronIndex: number;
}

interface Connection {
  source: string;
  target: string;
}

interface ModelMetadata {
  model_name: string;
  created_at: string;
  trainable_params: number;
  non_trainable_params: number;
  total_params: number;
}

interface OptimizerInfo {
  type: string;
  config: any;
}

interface ModelInfo {
  summary: string;
  layers: Layer[];
  optimizer: OptimizerInfo;
  loss: string;
  metrics: string[];
  metadata: ModelMetadata;
  weights: number[][][];
  biases: number[][];
}

export default function App() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [layers, setLayers] = useState<Layer[]>([]);
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedNeuron, setSelectedNeuron] = useState<Neuron | null>(null);
  const [hoveredConnection, setHoveredConnection] = useState<{source: string, target: string} | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const loadModel = async () => {
    if (!modelFile) return;
    const form = new FormData();
    form.append('file', modelFile);
    await api.post('/load_model', form);
    alert('Modelo cargado');
    fetchModelInfo();
  };

  const fetchModelInfo = async () => {
    const res = await api.get('/model_info');
    const info: ModelInfo = res.data;
    setModelInfo(info);
    setLayers(info.layers);
  };

  const buildNetwork = (layerData: Layer[]) => {
    if (!svgRef.current) return;
    const { clientWidth, clientHeight } = svgRef.current;

    // Espaciado horizontal para todas las capas (Dense, Dropout, etc.)
    const layerSpacing = clientWidth / (layerData.length + 1);

    const generatedNeurons: Neuron[] = [];
    const generatedConnections: Connection[] = [];

    // Genera neuronas solo para capas Dense/InputLayer
    layerData.forEach((layer, layerIdx) => {
      if (
        (layer.type === 'Dense' || layer.type === 'InputLayer') &&
        typeof layer.neuron_count === 'number' &&
        layer.neuron_count > 0
      ) {
        const count = layer.neuron_count;
        const neuronSpacing = clientHeight / (count + 1);

        for (let i = 0; i < count; i++) {
          generatedNeurons.push({
            id: `${layerIdx}-${i}`,
            x: (layerIdx + 1) * layerSpacing,
            y: (i + 1) * neuronSpacing,
            layerIndex: layerIdx,
            neuronIndex: i,
          });
        }
      }
    });

    // Conexiones solo entre capas Dense/InputLayer consecutivas
    // Busca los índices absolutos de las capas visualizables
    const visualLayerIndices = layerData
      .map((layer, idx) =>
        (layer.type === 'Dense' || layer.type === 'InputLayer') &&
        typeof layer.neuron_count === 'number' &&
        layer.neuron_count > 0
          ? idx
          : null
      )
      .filter(idx => idx !== null) as number[];

    for (let l = 0; l < visualLayerIndices.length - 1; l++) {
      const idxA = visualLayerIndices[l];
      const idxB = visualLayerIndices[l + 1];
      const countA = layerData[idxA].neuron_count!;
      const countB = layerData[idxB].neuron_count!;
      for (let i = 0; i < countA; i++) {
        for (let j = 0; j < countB; j++) {
          generatedConnections.push({
            source: `${idxA}-${i}`,
            target: `${idxB}-${j}`,
          });
        }
      }
    }

    setNeurons(generatedNeurons);
    setConnections(generatedConnections);
    console.log('Neurons:', generatedNeurons.length, 'Connections:', generatedConnections.length);
  };

  useEffect(() => {
    if (layers.length > 0) {
      setTimeout(() => {
        if (svgRef.current && svgRef.current.clientWidth > 0 && svgRef.current.clientHeight > 0) {
          buildNetwork(layers);
        }
      }, 100); // 100ms suele ser suficiente
    }
  }, [layers]);

  function CustomGraph() {
    const visibleConnections = selectedNeuron
      ? connections.filter(conn => conn.source === selectedNeuron.id || conn.target === selectedNeuron.id)
      : [];

    const inputLayer = Math.min(...neurons.map(n => n.layerIndex));
    const outputLayer = Math.max(...neurons.map(n => n.layerIndex));

    // Capas no Dense ni InputLayer
    const nonDenseLayers = layers
      .map((layer, idx) => ({ ...layer, idx }))
      .filter(layer => layer.type !== 'Dense' && layer.type !== 'InputLayer');

    // Para calcular la posición X de cada capa
    const layerSpacing = svgRef.current
      ? svgRef.current.clientWidth / (layers.length + 1)
      : 100;

    return (
      <svg ref={svgRef} width="100%" height="80vh">
        {/* Rectángulos para capas no Dense */}
        {nonDenseLayers.map((layer) => {
          const x = (layer.idx + 1) * layerSpacing - 30;
          const y = 40;
          return (
            <g key={layer.idx}>
              <rect
                x={x}
                y={y}
                width={60}
                height={svgRef.current ? svgRef.current.clientHeight - 80 : 200}
                fill="#ffe4b2"
                stroke="#b8860b"
                strokeWidth={2}
                rx={10}
                opacity={0.7}
              />
              <text
                x={x + 30}
                y={y + 30}
                textAnchor="middle"
                fontSize="14"
                fill="#b8860b"
                fontWeight="bold"
              >
                {layer.type}
              </text>
            </g>
          );
        })}

        {/* Solo conexiones de la neurona seleccionada */}
        {visibleConnections.map((conn, idx) => {
          const source = neurons.find(n => n.id === conn.source);
          const target = neurons.find(n => n.id === conn.target);
          if (!source || !target) return null;

          // Extraer índices de capa y neurona
          const [layerA, neuronA] = source.id.split('-').map(Number);
          const [layerB, neuronB] = target.id.split('-').map(Number);

          // Keras: weights[layerB] es [input_dim, output_dim]
          // Para la conexión de la neuronaA (origen, capa anterior) a la neuronaB (destino, capa actual)
          const weight = modelInfo?.weights?.[layerB]?.[neuronA]?.[neuronB];

          // Escala de gris: 220 (gris claro) a 0 (negro)
          let gray = 220;
          if (typeof weight === 'number') {
            const norm = Math.max(-2, Math.min(2, weight)); // recorta a [-2,2]
            const absNorm = Math.abs(norm) / 2; // [0,1]
            gray = Math.round(220 * (1 - absNorm)); // 220 (claro) a 0 (negro)
          }
          const color = `rgb(${gray},${gray},${gray})`;

          // Coordenadas para el cuadro del peso
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2 - 8;

          // ¿Está resaltada esta conexión?
          const isHighlighted =
            hoveredConnection &&
            hoveredConnection.source === conn.source &&
            hoveredConnection.target === conn.target;

          return (
            <g key={idx}>
              <line
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke={isHighlighted ? "#1976d2" : color}
                strokeWidth={isHighlighted ? 4 : 2}
                opacity={isHighlighted ? 1 : 0.8}
                onMouseEnter={() => setHoveredConnection({ source: conn.source, target: conn.target })}
                onMouseLeave={() => setHoveredConnection(null)}
                style={{ cursor: 'pointer' }}
              />
              {isHighlighted && weight !== undefined && (
                <>
                  {/* Cuadro de fondo para el peso */}
                  <rect
                    x={midX - 38}
                    y={midY - 18}
                    width={76}
                    height={28}
                    rx={6}
                    fill="#fff"
                    stroke="#1976d2"
                    strokeWidth={1.5}
                    opacity={0.95}
                  />
                  <text
                    x={midX}
                    y={midY}
                    fontSize="14"
                    fill="#1976d2"
                    textAnchor="middle"
                    alignmentBaseline="middle"
                    fontWeight="bold"
                    style={{ userSelect: 'none' }}
                  >
                    Peso: {Number(weight).toPrecision(4)}
                  </text>
                </>
              )}
            </g>
          );
        })}

        {/* Neuronas */}
        {neurons.map((neuron) => (
          <g key={neuron.id}>
            <circle
              cx={neuron.x}
              cy={neuron.y}
              r="10"
              fill={
                selectedNeuron?.id === neuron.id
                  ? '#88ccff'
                  : neuron.layerIndex === inputLayer
                  ? '#aaffaa'
                  : neuron.layerIndex === outputLayer
                  ? '#ffaaaa'
                  : '#cce5ff'
              }
              stroke="#333"
              onClick={() => setSelectedNeuron(neuron)}
            />
            {selectedNeuron?.id === neuron.id && (
              <text x={neuron.x + 12} y={neuron.y - 12} fontSize="12" fill="#333">
                Layer {neuron.layerIndex}, N{neuron.neuronIndex}
              </text>
            )}
          </g>
        ))}
      </svg>
    );
  }

  return (
    <div className="p-4">
      <Card>
        <CardContent>
          <h2 className="text-lg font-bold mb-2">Cargar modelo</h2>
          <Input type="file" onChange={e => setModelFile(e.target.files?.[0] || null)} />
          <Button onClick={loadModel} className="mt-2">Cargar</Button>
        </CardContent>
      </Card>

      {modelInfo && (
        <>
          <Card className="mt-4">
            <CardContent>
              <h2 className="text-lg font-bold mb-2">Visualización de Red Neuronal</h2>
              <CustomGraph />
              {selectedNeuron && (
                <div className="text-sm mt-4 bg-gray-50 p-2 rounded border">
                  <p><strong>Neurona seleccionada:</strong> Capa {selectedNeuron.layerIndex}, Índice {selectedNeuron.neuronIndex}</p>
                  {modelInfo.weights?.[selectedNeuron.layerIndex]?.[selectedNeuron.neuronIndex] && (
                    <>
                      <p className="mt-1">Pesos: <code>{JSON.stringify(modelInfo.weights[selectedNeuron.layerIndex][selectedNeuron.neuronIndex])}</code></p>
                      <p>Bias: <code>{modelInfo.biases[selectedNeuron.layerIndex][selectedNeuron.neuronIndex]}</code></p>
                    </>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="mt-4">
            <CardContent>
              <h2 className="text-lg font-bold mb-2">Detalles del Modelo</h2>
              <p className="text-sm mb-2">Nombre: <strong>{modelInfo.metadata.model_name}</strong></p>
              <p className="text-sm mb-2">Parámetros: <strong>{modelInfo.metadata.total_params}</strong> 
                (<span className="text-green-700">{modelInfo.metadata.trainable_params} entrenables</span>, 
                <span className="text-gray-700"> {modelInfo.metadata.non_trainable_params} no entrenables</span>)
              </p>
              <p className="text-sm mb-2">Función de pérdida: <strong>{modelInfo.loss || 'N/A'}</strong></p>
              <p className="text-sm mb-2">Optimizador: <strong>{modelInfo.optimizer?.type || 'N/A'}</strong></p>
              <p className="text-sm mb-4">Métricas: {modelInfo.metrics?.join(', ') || 'N/A'}</p>

              <ul className="text-sm list-disc pl-5">
                {modelInfo.layers.map((layer, idx) => (
                  <li key={idx} className="mb-2">
                    <strong>{layer.name}</strong> ({layer.type || 'Capa'}): <br />
                    - Activación: {layer.activation || 'N/A'}<br />
                    - Salida: {layer.output_shape?.join(', ') || 'N/A'}<br />
                    - Config: {layer.config ? JSON.stringify(layer.config) : 'N/A'}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
