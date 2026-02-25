// NNV - Neural Network Visualizer
// Aplicación para visualizar y analizar modelos de redes neuronales Keras

import { useState, useEffect, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

// === INTERFACES ===

interface Layer {
  index: number;
  name: string;
  output_shape?: number[];
  input_shape?: number[];
  activation?: string | null;
  type?: string;
  config?: any;
  neuron_count?: number;
  trainable?: boolean;
  dtype?: string;
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
  keras_version?: string;
  backend?: string;
  date_saved?: string;
}

interface OptimizerInfo {
  type: string;
  config: any;
}

interface KerasFileStructure {
  files: string[];
  config_json?: any;
  metadata_json?: any;
}

interface WeightTensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
  min_value: number;
  max_value: number;
  mean_value: number;
  std_value: number;
}

interface LayerWeightsInfo {
  layer_name: string;
  tensors: WeightTensorInfo[];
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
  keras_file_structure?: KerasFileStructure;
  detailed_weights?: LayerWeightsInfo[];
}

// Tabs disponibles
type TabType = 'file' | 'model' | 'params' | 'visualization';

export default function App() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [npyFile, setNpyFile] = useState<File | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [layers, setLayers] = useState<Layer[]>([]);
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedNeuron, setSelectedNeuron] = useState<Neuron | null>(null);
  const [hoveredConnection, setHoveredConnection] = useState<{source: string, target: string} | null>(null);
  const [hoveredNeuron, setHoveredNeuron] = useState<Neuron | null>(null);
  const [selectedLayerIdx, setSelectedLayerIdx] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('file');
  const svgRef = useRef<SVGSVGElement>(null);

  const loadModel = async () => {
    if (!modelFile) return;
    const form = new FormData();
    form.append('file', modelFile);
    await api.post('/load_model', form);
    fetchModelInfo();
  };

  const uploadNpy = async () => {
    if (!npyFile) return;
    const form = new FormData();
    form.append('npy', npyFile);
    await api.post('/carga_parametros', form);
  };

  const fetchModelInfo = async () => {
    const res = await api.get('/model_info');
    const info: ModelInfo = res.data;
    setModelInfo(info);
    setLayers(info.layers);
  };

  const buildNetwork = (layerData: Layer[]) => {
    if (!svgRef.current) return;
    
    const clientWidth = svgRef.current.clientWidth > 0 ? svgRef.current.clientWidth : 1200;
    const clientHeight = svgRef.current.clientHeight > 0 ? svgRef.current.clientHeight : 600;
    const layerSpacing = clientWidth / (layerData.length + 1);

    const generatedNeurons: Neuron[] = [];
    const generatedConnections: Connection[] = [];
    const verticalMargin = 40;
    
    // Generar neuronas para capas Dense e InputLayer
    layerData.forEach((layer, layerIdx) => {
      if (
        (layer.type === 'Dense' || layer.type === 'InputLayer') &&
        typeof layer.neuron_count === 'number' &&
        layer.neuron_count > 0
      ) {
        const count = layer.neuron_count;
        const neuronSpacing = (clientHeight - verticalMargin * 2) / (count - 1 || 1);

        for (let i = 0; i < count; i++) {
          generatedNeurons.push({
            id: `${layerIdx}-${i}`,
            x: (layerIdx + 1) * layerSpacing,
            y: verticalMargin + i * neuronSpacing,
            layerIndex: layerIdx,
            neuronIndex: i,
          });
        }
      }
    });

    // Generar conexiones entre capas consecutivas
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
  };

  useEffect(() => {
    if (layers.length > 0 && activeTab === 'visualization') {
      // Pequeño delay para asegurar que el SVG esté renderizado
      setTimeout(() => {
        if (svgRef.current && svgRef.current.clientWidth > 0 && svgRef.current.clientHeight > 0) {
          buildNetwork(layers);
        } else {
          // Reintentar si el SVG no tiene dimensiones aún
          setTimeout(() => {
            if (svgRef.current && svgRef.current.clientWidth > 0) {
              buildNetwork(layers);
            }
          }, 200);
        }
      }, 100);
    }
  }, [layers, activeTab]);

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
      <svg ref={svgRef} width="100%" height="600" style={{ minHeight: '600px', background: '#fafafa' }}>
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

        {/* Conexiones de la neurona seleccionada */}
        {visibleConnections.map((conn, idx) => {
          const source = neurons.find(n => n.id === conn.source);
          const target = neurons.find(n => n.id === conn.target);
          if (!source || !target) return null;

          const [, neuronA] = source.id.split('-').map(Number);
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
                    Peso: {formatNumber(weight, 4)}
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
              onClick={() => {
                // Toggle: si ya está seleccionada, deseleccionar
                if (selectedNeuron?.id === neuron.id) {
                  setSelectedNeuron(null);
                } else {
                  setSelectedNeuron(neuron);
                }
              }}
              onMouseEnter={() => setHoveredNeuron(neuron)}
              onMouseLeave={() => setHoveredNeuron(null)}
              style={{ cursor: 'pointer', filter: hoveredNeuron?.id === neuron.id ? 'drop-shadow(0 0 6px rgba(0,0,0,0.5))' : 'none' }}
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

  // Componente de mini previsualización para la tab de Modelo
  function MiniNetworkPreview({ highlightLayerIdx }: { highlightLayerIdx: number | null }) {
    // Ancho dinámico: mínimo 50px por capa, mínimo total 200px
    const layerSpacing = 50;
    const previewWidth = Math.max(200, (layers.length + 1) * layerSpacing);
    const previewHeight = 200;
    const verticalMargin = 20;

    return (
      <svg width={previewWidth} height={previewHeight} style={{ background: '#f8f9fa', borderRadius: '8px', border: '1px solid #ddd' }}>
        {/* Capas no Dense (BatchNorm, Dropout, etc) */}
        {layers.map((layer, idx) => {
          if (layer.type === 'Dense' || layer.type === 'InputLayer') return null;
          const x = (idx + 1) * layerSpacing - 15;
          return (
            <g key={`nondense-${idx}`}>
              <rect
                x={x}
                y={verticalMargin}
                width={30}
                height={previewHeight - verticalMargin * 2}
                fill={highlightLayerIdx === idx ? '#ffcc80' : '#ffe4b2'}
                stroke={highlightLayerIdx === idx ? '#e67e22' : '#b8860b'}
                strokeWidth={highlightLayerIdx === idx ? 3 : 1}
                rx={5}
                opacity={0.8}
              />
              <text x={x + 15} y={verticalMargin + 15} textAnchor="middle" fontSize="8" fill="#b8860b">
                {(layer.type || '').substring(0, 4)}
              </text>
            </g>
          );
        })}

        {/* Neuronas para capas Dense/InputLayer */}
        {layers.map((layer, layerIdx) => {
          if (layer.type !== 'Dense' && layer.type !== 'InputLayer') return null;
          if (!layer.neuron_count || layer.neuron_count === 0) return null;
          
          const count = Math.min(layer.neuron_count, 10); // Limitar a 10 para la mini vista
          const neuronSpacing = (previewHeight - verticalMargin * 2) / (count - 1 || 1);
          const isHighlighted = highlightLayerIdx === layerIdx;
          
          return (
            <g key={`layer-${layerIdx}`}>
              {/* Rectángulo de resaltado */}
              {isHighlighted && (
                <rect
                  x={(layerIdx + 1) * layerSpacing - 20}
                  y={verticalMargin - 10}
                  width={40}
                  height={previewHeight - verticalMargin * 2 + 20}
                  fill="none"
                  stroke="#e67e22"
                  strokeWidth={3}
                  rx={8}
                />
              )}
              {/* Neuronas */}
              {Array.from({ length: count }).map((_, i) => (
                <circle
                  key={`${layerIdx}-${i}`}
                  cx={(layerIdx + 1) * layerSpacing}
                  cy={verticalMargin + i * neuronSpacing}
                  r={5}
                  fill={isHighlighted ? '#ffcc80' : '#cce5ff'}
                  stroke={isHighlighted ? '#e67e22' : '#333'}
                  strokeWidth={isHighlighted ? 2 : 1}
                />
              ))}
              {/* Indicador de más neuronas */}
              {layer.neuron_count > 10 && (
                <text
                  x={(layerIdx + 1) * layerSpacing}
                  y={previewHeight - 5}
                  textAnchor="middle"
                  fontSize="8"
                  fill="#666"
                >
                  +{layer.neuron_count - 10}
                </text>
              )}
            </g>
          );
        })}
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
  // Formatea un valor numérico de forma segura. Devuelve 'N/A' si no es número finito.
  function formatNumber(value: any, precision = 5) {
    const n = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(n)) return 'N/A';
    try {
      return n.toPrecision(precision);
    } catch {
      return String(n);
    }
  }

  // Calcular estadísticas a partir de una estructura que puede contener
  // números, arrays anidados, o valores undefined. Recoge recursivamente números finitos.
  function getStats(arr: any) {
    if (arr == null) return null;
    const nums: number[] = [];
    const collect = (x: any) => {
      if (x == null) return;
      if (Array.isArray(x)) {
        x.forEach(collect);
        return;
      }
      const n = Number(x);
      if (Number.isFinite(n)) nums.push(n);
    };
    collect(arr);
    if (nums.length === 0) return null;
    nums.sort((a, b) => a - b);
    const min = nums[0];
    const max = nums[nums.length - 1];
    const mean = nums.reduce((s, v) => s + v, 0) / nums.length;
    const median =
      nums.length % 2 === 0
        ? (nums[nums.length / 2 - 1] + nums[nums.length / 2]) / 2
        : nums[Math.floor(nums.length / 2)];
    const variance = nums.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / nums.length;
    return { min, max, mean, median, variance };
  }

  // Tab button component
  const TabButton = ({ tab, label, icon }: { tab: TabType; label: string; icon: string }) => (
    <button
      onClick={() => setActiveTab(tab)}
      className={`px-4 py-2 font-semibold text-sm rounded-t-lg transition-all ${
        activeTab === tab
          ? 'bg-white border-t-2 border-l border-r border-blue-500 text-blue-700 -mb-px'
          : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-transparent'
      }`}
    >
      {icon} {label}
    </button>
  );

  return (
    <div className="p-4">
      <Card>
        <CardContent>
          <h2 className="text-lg font-bold mb-2">Cargar modelo</h2>
          <div className="flex flex-col gap-2">
            <Input
              type="file"
              accept=".keras"
              onChange={e => {
                const file = e.target.files?.[0];
                if (file && file.name.endsWith('.keras')) setModelFile(file);
                else setModelFile(null);
              }}
            />
            <Button
              onClick={loadModel}
              className="mt-2"
              disabled={!modelFile}
            >
              Cargar modelo
            </Button>
            <Input
              type="file"
              accept=".npy"
              onChange={e => {
                const file = e.target.files?.[0];
                if (file && file.name.endsWith('.npy')) setNpyFile(file);
                else setNpyFile(null);
              }}
            />
            <Button
              onClick={uploadNpy}
              className="mt-2"
              disabled={!npyFile}
            >
              Cargar parámetros
            </Button>
          </div>
        </CardContent>
      </Card>

      {modelInfo && (
        <>
          {/* Tab Navigation */}
          <div className="mt-4 flex gap-1 border-b border-gray-300">
            <TabButton tab="file" label="Archivo" icon="📦" />
            <TabButton tab="model" label="Modelo" icon="🧠" />
            <TabButton tab="params" label="Parámetros" icon="⚖️" />
            <TabButton tab="visualization" label="Visualización" icon="🔗" />
          </div>

          {/* Tab: Archivo - Estructura del archivo .keras */}
          {activeTab === 'file' && modelInfo.keras_file_structure && (
            <Card className="mt-0 border-2 border-purple-300 bg-purple-50 rounded-t-none">
              <CardContent>
                <h2 className="text-lg font-bold mb-2 text-purple-800">📦 Estructura del archivo .keras</h2>
                <p className="text-sm mb-2">
                  El archivo <code>.keras</code> es un ZIP que contiene:
                </p>
                <div className="flex flex-wrap gap-2 mb-3">
                  {modelInfo.keras_file_structure.files.map((file, idx) => (
                    <span
                      key={idx}
                      className={`px-2 py-1 rounded text-xs font-mono ${
                        file.endsWith('.json') ? 'bg-blue-100 text-blue-800' :
                        file.endsWith('.h5') ? 'bg-green-100 text-green-800' :
                        'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {file}
                    </span>
                  ))}
                </div>
                {modelInfo.keras_file_structure.metadata_json && (
                  <div className="text-sm bg-white p-2 rounded border mb-2">
                    <strong>Metadatos del archivo:</strong>
                    <ul className="ml-4 list-disc text-xs mt-1">
                      {Object.entries(modelInfo.keras_file_structure.metadata_json).map(([key, value]) => (
                        <li key={key}>{key}: <code>{String(value)}</code></li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Tab: Modelo - Detalles del modelo y capas */}
          {activeTab === 'model' && (
            <>
              <Card className="mt-0 rounded-t-none">
                <CardContent>
                  <h2 className="text-lg font-bold mb-2">Detalles del Modelo</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm mb-2">
                    Nombre: <strong>{modelInfo.metadata.model_name}</strong>
                  </p>
                  <p className="text-sm mb-2">
                    Parámetros: <strong>{modelInfo.metadata.total_params.toLocaleString()}</strong> (
                    <span className="text-green-700">{modelInfo.metadata.trainable_params.toLocaleString()} entrenables</span>, 
                    <span className="text-gray-700">{modelInfo.metadata.non_trainable_params.toLocaleString()} no entrenables</span>)
                  </p>
                  <p className="text-sm mb-2">
                    Función de pérdida: <strong>{modelInfo.loss || 'N/A'}</strong>
                  </p>
                  <p className="text-sm mb-2">
                    Optimizador: <strong>{modelInfo.optimizer?.type || 'N/A'}</strong>
                  </p>
                  <p className="text-sm mb-4">
                    Métricas: <strong>{modelInfo.metrics?.join(', ') || 'N/A'}</strong>
                  </p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-sm font-semibold mb-2">Información del archivo .keras:</p>
                  {modelInfo.metadata.keras_version && (
                    <p className="text-xs mb-1">Keras: <code>{modelInfo.metadata.keras_version}</code></p>
                  )}
                  {modelInfo.metadata.backend && (
                    <p className="text-xs mb-1">Backend: <code>{modelInfo.metadata.backend}</code></p>
                  )}
                  {modelInfo.metadata.date_saved && (
                    <p className="text-xs mb-1">Guardado: <code>{modelInfo.metadata.date_saved}</code></p>
                  )}
                  {!modelInfo.metadata.keras_version && !modelInfo.metadata.backend && (
                    <p className="text-xs text-gray-500">No hay metadatos adicionales disponibles</p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Lista horizontal de capas y detalle de capa seleccionada */}
          <Card className="mt-4">
            <CardContent>
              <h2 className="text-lg font-bold mb-2">Capas del Modelo</h2>
              <div className="flex flex-wrap gap-2 mb-4">
                {modelInfo.layers.map((layer, idx) => (
                  <button
                    key={idx}
                    className={`px-3 py-1 rounded border text-sm font-semibold transition ${
                      selectedLayerIdx === idx
                        ? 'bg-blue-200 border-blue-600 text-blue-900'
                        : layer.type === 'Dense' ? 'bg-blue-50 border-blue-300 text-blue-700 hover:bg-blue-100'
                        : layer.type === 'InputLayer' ? 'bg-green-50 border-green-300 text-green-700 hover:bg-green-100'
                        : layer.type === 'Dropout' ? 'bg-yellow-50 border-yellow-300 text-yellow-700 hover:bg-yellow-100'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                    }`}
                    onClick={() => setSelectedLayerIdx(selectedLayerIdx === idx ? null : idx)}
                  >
                    {layer.name}
                    <span className="ml-1 text-xs opacity-60">({layer.type})</span>
                  </button>
                ))}
              </div>
              
              {/* Grid con detalles y mini previsualización */}
              {selectedLayerIdx !== null && modelInfo.layers[selectedLayerIdx] && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  {/* Detalles de la capa (2/3) */}
                  <Card className="lg:col-span-2 border-4 rounded-lg border-blue-600 bg-blue-50">
                    <CardContent>
                    <div className="flex justify-between items-start">
                      <div>
                        <strong className="text-lg">{modelInfo.layers[selectedLayerIdx].name}</strong>{' '}
                        <span className="px-2 py-0.5 bg-blue-200 rounded text-xs text-blue-800">
                          {modelInfo.layers[selectedLayerIdx].type || 'Capa'}
                        </span>
                      </div>
                      <div className="text-right text-xs">
                        {modelInfo.layers[selectedLayerIdx].trainable !== undefined && (
                          <span className={`px-2 py-0.5 rounded ${
                            modelInfo.layers[selectedLayerIdx].trainable 
                              ? 'bg-green-200 text-green-800' 
                              : 'bg-gray-200 text-gray-600'
                          }`}>
                            {modelInfo.layers[selectedLayerIdx].trainable ? '✓ Entrenable' : '✗ No entrenable'}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="text-sm">
                        <p className="font-semibold mb-1">Dimensiones:</p>
                        {modelInfo.layers[selectedLayerIdx].input_shape && (
                          <div className="text-xs mb-1">
                            Entrada: <code className="bg-white px-1 rounded">[{modelInfo.layers[selectedLayerIdx].input_shape.join(', ')}]</code>
                          </div>
                        )}
                        {modelInfo.layers[selectedLayerIdx].output_shape && (
                          <div className="text-xs mb-1">
                            Salida: <code className="bg-white px-1 rounded">[{modelInfo.layers[selectedLayerIdx].output_shape.join(', ')}]</code>
                          </div>
                        )}
                        {modelInfo.layers[selectedLayerIdx].neuron_count && (
                          <div className="text-xs mb-1">
                            Neuronas: <strong>{modelInfo.layers[selectedLayerIdx].neuron_count}</strong>
                          </div>
                        )}
                        {modelInfo.layers[selectedLayerIdx].activation && 
                         modelInfo.layers[selectedLayerIdx].activation !== 'None' && (
                          <div className="text-xs mb-1">
                            Activación: <code className="bg-yellow-100 px-1 rounded">{modelInfo.layers[selectedLayerIdx].activation}</code>
                          </div>
                        )}
                        {modelInfo.layers[selectedLayerIdx].dtype && (
                          <div className="text-xs mb-1">
                            Dtype: <code className="bg-gray-100 px-1 rounded">{modelInfo.layers[selectedLayerIdx].dtype}</code>
                          </div>
                        )}
                      </div>
                      <div className="text-sm">
                        <p className="font-semibold mb-1">Configuración completa:</p>
                        {modelInfo.layers[selectedLayerIdx].config && (
                          <div className="text-xs bg-white p-2 rounded border max-h-40 overflow-y-auto">
                            {renderConfig(modelInfo.layers[selectedLayerIdx].config)}
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                  {/* Mini previsualización (1/3) */}
                  <Card className="border-2 rounded-lg border-orange-400 bg-orange-50">
                    <CardContent>
                      <p className="font-semibold mb-2 text-sm text-orange-800">📍 Ubicación en la red</p>
                      <div className="overflow-x-auto">
                        <MiniNetworkPreview highlightLayerIdx={selectedLayerIdx} />
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </CardContent>
          </Card>
            </>
          )}

          {/* Tab: Parámetros - Estadísticas de pesos */}
          {activeTab === 'params' && (
            <>
              {modelInfo.detailed_weights && modelInfo.detailed_weights.length > 0 && (
                <Card className="mt-0 border-2 border-green-300 bg-green-50 rounded-t-none">
                  <CardContent>
                    <h2 className="text-lg font-bold mb-2 text-green-800">⚖️ Estadísticas de Pesos por Capa</h2>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="bg-green-100">
                            <th className="px-2 py-1 text-left">Capa</th>
                            <th className="px-2 py-1 text-left">Tensor</th>
                            <th className="px-2 py-1 text-right">Shape</th>
                            <th className="px-2 py-1 text-right">Params</th>
                            <th className="px-2 py-1 text-right">Min</th>
                            <th className="px-2 py-1 text-right">Max</th>
                            <th className="px-2 py-1 text-right">Media</th>
                            <th className="px-2 py-1 text-right">Std</th>
                          </tr>
                        </thead>
                        <tbody>
                          {modelInfo.detailed_weights.map((layerWeight, idx) => (
                            layerWeight.tensors.map((tensor, tIdx) => (
                              <tr key={`${idx}-${tIdx}`} className={idx % 2 === 0 ? 'bg-white' : 'bg-green-50'}>
                                {tIdx === 0 && (
                                  <td className="px-2 py-1 font-semibold" rowSpan={layerWeight.tensors.length}>
                                    {layerWeight.layer_name}
                                  </td>
                                )}
                                <td className="px-2 py-1 font-mono text-gray-600">
                                  {tensor.name.split('/').pop()}
                                </td>
                                <td className="px-2 py-1 text-right font-mono">
                                  [{tensor.shape.join(', ')}]
                                </td>
                                <td className="px-2 py-1 text-right">
                                  {tensor.size.toLocaleString()}
                                </td>
                                <td className="px-2 py-1 text-right font-mono">
                                  {tensor.min_value.toFixed(4)}
                                </td>
                                <td className="px-2 py-1 text-right font-mono">
                                  {tensor.max_value.toFixed(4)}
                                </td>
                                <td className="px-2 py-1 text-right font-mono">
                                  {tensor.mean_value.toFixed(4)}
                                </td>
                                <td className="px-2 py-1 text-right font-mono">
                                  {tensor.std_value.toFixed(4)}
                                </td>
                              </tr>
                            ))
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {/* Tab: Visualización - Red neuronal */}
          {activeTab === 'visualization' && (
            <>
              <Card className="mt-0 rounded-t-none">
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
                                  <li>Mínimo: {formatNumber(stats.min, 5)}</li>
                                  <li>Máximo: {formatNumber(stats.max, 5)}</li>
                                  <li>Media: {formatNumber(stats.mean, 5)}</li>
                                  <li>Mediana: {formatNumber(stats.median, 5)}</li>
                                  <li>Varianza: {formatNumber(stats.variance, 5)}</li>
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
                                <li>Mínimo: {formatNumber(stats.min, 5)}</li>
                                <li>Máximo: {formatNumber(stats.max, 5)}</li>
                                <li>Media: {formatNumber(stats.mean, 5)}</li>
                                <li>Mediana: {formatNumber(stats.median, 5)}</li>
                                <li>Varianza: {formatNumber(stats.variance, 5)}</li>
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
                                    <li>Mínimo: {formatNumber(outStats.min, 5)}</li>
                                    <li>Máximo: {formatNumber(outStats.max, 5)}</li>
                                    <li>Media: {formatNumber(outStats.mean, 5)}</li>
                                    <li>Mediana: {formatNumber(outStats.median, 5)}</li>
                                    <li>Varianza: {formatNumber(outStats.variance, 5)}</li>
                                  </>
                                ) : (
                                  <li>No hay estadísticas disponibles.</li>
                                )}
                              </ul>
                              <p>Estadísticas de pesos de entrada:</p>
                              <ul className="ml-4 list-disc">
                                {inStats ? (
                                  <>
                                    <li>Mínimo: {formatNumber(inStats.min, 5)}</li>
                                    <li>Máximo: {formatNumber(inStats.max, 5)}</li>
                                    <li>Media: {formatNumber(inStats.mean, 5)}</li>
                                    <li>Mediana: {formatNumber(inStats.median, 5)}</li>
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
            </>
          )}
        </>
      )}
    </div>
  );
}