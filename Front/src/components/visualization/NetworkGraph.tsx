import { Layer, Neuron, Connection, ModelInfo } from "../../types";
import { formatNumber } from "../../utils/modelHelpers";
import { RefObject } from "react";

interface NetworkGraphProps {
  layers: Layer[];
  neurons: Neuron[];
  connections: Connection[];
  modelInfo: ModelInfo | null;
  selectedNeuron: Neuron | null;
  setSelectedNeuron: (neuron: Neuron | null) => void;
  hoveredConnection: { source: string; target: string } | null;
  setHoveredConnection: (conn: { source: string; target: string } | null) => void;
  hoveredNeuron: Neuron | null;
  setHoveredNeuron: (neuron: Neuron | null) => void;
  svgRef: RefObject<SVGSVGElement | null>;
}

export function NetworkGraph({
  layers,
  neurons,
  connections,
  modelInfo,
  selectedNeuron,
  setSelectedNeuron,
  hoveredConnection,
  setHoveredConnection,
  hoveredNeuron,
  setHoveredNeuron,
  svgRef,
}: NetworkGraphProps) {
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
    <svg ref={svgRef as any} width="100%" height="600" style={{ minHeight: '600px', background: '#fafafa' }}>
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
