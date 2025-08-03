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
  const [hoveredNeuron, setHoveredNeuron] = useState<Neuron | null>(null);
  const [selectedLayerIdx, setSelectedLayerIdx] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const loadModel = async () => {
    if (!modelFile) return;
    const form = new FormData();
    form.append('file', modelFile);
    await api.post('/load_model', form);
    // alert('Modelo cargado');
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
      <svg ref={svgRef} width="100%" height="90vh">
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

          const [layerA, neuronA] = source.id.split('-').map(Number);
          const [layerB, neuronB] = target.id.split('-').map(Number);

          const weight = modelInfo?.weights?.[layerB]?.[neuronA]?.[neuronB];

          let gray = 220;
          if (typeof weight === 'number') {
            const norm = Math.max(-2, Math.min(2, weight));
            const absNorm = Math.abs(norm) / 2;
            gray = Math.round(220 * (1 - absNorm));
          }
          const color = `rgb(${gray},${gray},${gray})`;

          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2 - 8;

          // Solo resalta si está en hover la conexión, o si el mouse está sobre la neurona de destino Y la neurona de origen es la seleccionada
          const isHighlighted =
            (hoveredConnection &&
              hoveredConnection.source === conn.source &&
              hoveredConnection.target === conn.target) ||
            (
              hoveredNeuron &&
              (hoveredNeuron.id === conn.source || hoveredNeuron.id === conn.target) &&
              (!selectedNeuron || hoveredNeuron.id !== selectedNeuron.id)
            );

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
              onMouseEnter={() => setHoveredNeuron(neuron)}
              onMouseLeave={() => setHoveredNeuron(null)}
              style={{ cursor: 'pointer' }}
            />
            {selectedNeuron?.id === neuron.id && (
              <text x={neuron.x + 12} y={neuron.y - 12} fontSize="12" fill="#333">
                Layer {neuron.layerIndex}, N{neuron.neuronIndex}
              </text>
            )}
          </g>
        ))}

        {/* Resaltado de la capa seleccionada */}
        {selectedLayerIdx !== null && (
          (() => {
            const layer = layers[selectedLayerIdx];
            if (!layer) return null;
            const layerSpacing = svgRef.current
              ? svgRef.current.clientWidth / (layers.length + 1)
              : 100;
            const x = (selectedLayerIdx + 1) * layerSpacing - 30;
            const width = 60;
            const svgHeight = svgRef.current ? svgRef.current.clientHeight : 400;
            const y = 0;
            const height = svgHeight;

            if (layer.type === 'Dense') {
              // Dibuja un óvalo para Dense
              return (
                <ellipse
                  cx={x + width / 2}
                  cy={y + height / 2}
                  rx={width / 2}
                  ry={height + 0.5 / 2}
                  fill="none"
                  stroke="#e67e22"
                  strokeWidth={4}
                  opacity={0.95}
                  style={{ pointerEvents: 'none' }}
                />
              );
            } else {
              // Rectángulo para capas no Dense
              return (
                <rect
                  x={x}
                  y={40}
                  width={width}
                  height={svgHeight - 80}
                  fill="none"
                  stroke="#e67e22"
                  strokeWidth={4}
                  rx={10}
                  opacity={0.95}
                  style={{ pointerEvents: 'none' }}
                />
              );
            }
          })()
        )}
      </svg>
    );
  }

  // Función recursiva para mostrar objetos como listas anidadas
  function renderConfig(config: any) {
    if (typeof config !== 'object' || config === null) {
      return <span>{String(config)}</span>;
    }
    if (Array.isArray(config)) {
      return (
        <ul className="ml-4 list-disc">
          {config.map((item, idx) => (
            <li key={idx}>{renderConfig(item)}</li>
          ))}
        </ul>
      );
    }
    return (
      <ul className="ml-4 list-disc">
        {Object.entries(config).map(([key, value]) => (
          <li key={key}>
            {key}: {typeof value === 'object' && value !== null ? renderConfig(value) : String(value)}
          </li>
        ))}
      </ul>
    );
  }

  // Función para calcular estadísticas
  function getStats(arr: number[]) {
    if (!arr || arr.length === 0) return null;
    const sorted = [...arr].sort((a, b) => a - b);
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const median =
      arr.length % 2 === 0
        ? (sorted[arr.length / 2 - 1] + sorted[arr.length / 2]) / 2
        : sorted[Math.floor(arr.length / 2)];
    const variance =
      arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length;
    return { min, max, mean, median, variance };
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
                  {/* Estadísticas de Input (primer capa) */}
                  {selectedNeuron.layerIndex === Math.min(...neurons.map(n => n.layerIndex)) ? (
                    <div className="mt-1">
                      <p>Estadísticas de pesos de entrada:</p>
                      <ul className="ml-4 list-disc">
                        <li>Sin pesos de entrada</li>
                      </ul>
                    </div>
                  ) : (
                    modelInfo.weights?.[selectedNeuron.layerIndex] && (
                      (() => {
                        // Pesos de entrada: columna de la neurona en la matriz de la capa actual
                        const weightsArr = modelInfo.weights[selectedNeuron.layerIndex].map(row => row[selectedNeuron.neuronIndex]);
                        const stats = getStats(weightsArr);
                        return stats ? (
                          <div className="mt-1">
                            <p>Estadísticas de pesos de entrada:</p>
                            <ul className="ml-4 list-disc">
                              <li>Mínimo: {stats.min.toPrecision(5)}</li>
                              <li>Máximo: {stats.max.toPrecision(5)}</li>
                              <li>Media: {stats.mean.toPrecision(5)}</li>
                              <li>Mediana: {stats.median.toPrecision(5)}</li>
                              <li>Varianza: {stats.variance.toPrecision(5)}</li>
                            </ul>
                          </div>
                        ) : null;
                      })()
                    )
                  )}
                  {/* Estadísticas de Output (última capa) */}
                  {selectedNeuron.layerIndex === Math.max(...neurons.map(n => n.layerIndex)) && modelInfo.weights?.[selectedNeuron.layerIndex] && (
                    (() => {
                      // Pesos de salida: fila de la neurona en la matriz de la capa actual
                      const weightsArr = modelInfo.weights[selectedNeuron.layerIndex][selectedNeuron.neuronIndex];
                      const stats = getStats(weightsArr);
                      return stats ? (
                        <div className="mt-1">
                          <p>Estadísticas de pesos de salida:</p>
                          <ul className="ml-4 list-disc">
                            <li>Mínimo: {stats.min.toPrecision(5)}</li>
                            <li>Máximo: {stats.max.toPrecision(5)}</li>
                            <li>Media: {stats.mean.toPrecision(5)}</li>
                            <li>Mediana: {stats.median.toPrecision(5)}</li>
                            <li>Varianza: {stats.variance.toPrecision(5)}</li>
                          </ul>
                        </div>
                      ) : null;
                    })()
                  )}
                  {/* Estadísticas de pesos internos (capas ocultas Dense) */}
                  {selectedNeuron.layerIndex !== Math.min(...neurons.map(n => n.layerIndex)) &&
                    selectedNeuron.layerIndex !== Math.max(...neurons.map(n => n.layerIndex)) &&
                    modelInfo.weights?.[selectedNeuron.layerIndex] && modelInfo.weights?.[selectedNeuron.layerIndex + 1] && (
                    (() => {
                      // Pesos de salida (fila actual)
                      const outArr = modelInfo.weights[selectedNeuron.layerIndex][selectedNeuron.neuronIndex];
                      // Pesos de entrada (columna en la matriz de la siguiente capa)
                      const inArr = modelInfo.weights[selectedNeuron.layerIndex + 1].map(row => row[selectedNeuron.neuronIndex]);
                      const outStats = getStats(outArr);
                      const inStats = getStats(inArr);
                      return (
                        <div className="mt-1">
                          <p>Estadísticas de pesos de salida:</p>
                          <ul className="ml-4 list-disc">
                            {outStats ? (
                              <>
                                <li>Mínimo: {outStats.min.toPrecision(5)}</li>
                                <li>Máximo: {outStats.max.toPrecision(5)}</li>
                                <li>Media: {outStats.mean.toPrecision(5)}</li>
                                <li>Mediana: {outStats.median.toPrecision(5)}</li>
                                <li>Varianza: {outStats.variance.toPrecision(5)}</li>
                              </>
                            ) : (
                              <li>No hay estadísticas disponibles.</li>
                            )}
                          </ul>
                          <p>Estadísticas de pesos de entrada:</p>
                          <ul className="ml-4 list-disc">
                            {inStats ? (
                              <>
                                <li>Mínimo: {inStats.min.toPrecision(5)}</li>
                                <li>Máximo: {inStats.max.toPrecision(5)}</li>
                                <li>Media: {inStats.mean.toPrecision(5)}</li>
                                <li>Mediana: {inStats.median.toPrecision(5)}</li>
                                <li>Varianza: {inStats.variance.toPrecision(5)}</li>
                              </>
                            ) : (
                              <li>No hay estadísticas disponibles.</li>
                            )}
                          </ul>
                        </div>
                      );
                    })()
                  )}
                  {/* Bias */}
                  {modelInfo.biases?.[selectedNeuron.layerIndex]?.[selectedNeuron.neuronIndex] !== undefined && (
                    <p>Bias: <code>{modelInfo.biases[selectedNeuron.layerIndex][selectedNeuron.neuronIndex]}</code></p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="mt-4">
            <CardContent>
              <h2 className="text-lg font-bold mb-2">Detalles del Modelo</h2>
              <p className="text-sm mb-2">Nombre: <strong>{modelInfo.metadata.model_name}</strong></p>
              <p className="text-sm mb-2">Parámetros: <strong>{modelInfo.metadata.total_params}</strong> (<span className="text-green-700">{modelInfo.metadata.trainable_params} entrenables</span>, <span className="text-gray-700"> {modelInfo.metadata.non_trainable_params} no entrenables</span>)</p>
              <p className="text-sm mb-2">Función de pérdida: <strong>{modelInfo.loss || 'N/A'}</strong></p>
              <p className="text-sm mb-2">Optimizador: <strong>{modelInfo.optimizer?.type || 'N/A'}</strong></p>
              <p className="text-sm mb-4">Métricas: <strong>{modelInfo.metrics?.join(', ') || 'N/A'}</strong></p>
            </CardContent>
          </Card>

          {/* Grilla de capas fuera del Card de detalles */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-4">
            {modelInfo.layers.map((layer, idx) => (
              <Card
                key={idx}
                className={`mb-2 cursor-pointer border-4 rounded-lg`}
                onClick={() => setSelectedLayerIdx(selectedLayerIdx === idx ? null : idx)}
                style={
                  selectedLayerIdx === idx
                    ? {
                        borderColor: '#2563eb',
                        boxShadow: '0 4px 24px 0 rgba(37,99,235,0.15)',
                        borderWidth: 2,
                      }
                    : {

                      }
                }
              >
                <CardContent>
                  <strong>{layer.name}</strong> <span className="text-xs text-gray-500">({layer.type || 'Capa'})</span>
                  <div className="mt-2 text-sm">
                    <div>- Activación: <strong>{layer.activation || 'N/A'}</strong></div>
                    <div>- Salida: {layer.output_shape?.join(', ') || 'N/A'}</div>
                    <div>- Config: {layer.config ? renderConfig(layer.config) : 'N/A'}</div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
