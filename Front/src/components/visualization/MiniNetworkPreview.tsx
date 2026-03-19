// src/components/visualization/MiniNetworkPreview.tsx
import { Layer } from "../../types";

interface MiniNetworkPreviewProps {
  layers: Layer[];
  highlightLayerIdx: number | null;
}

export function MiniNetworkPreview({ layers, highlightLayerIdx }: MiniNetworkPreviewProps) {
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
