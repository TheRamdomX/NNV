import { Card, CardContent } from "@/components/ui/card";
import { ModelInfo, Neuron, Connection, Layer } from "../../types";
import { NetworkGraph } from "../visualization/NetworkGraph";
import { getStats, formatNumber } from "../../utils/modelHelpers";
import { RefObject } from "react";

interface VisualizationTabProps {
  layers: Layer[];
  neurons: Neuron[];
  connections: Connection[];
  modelInfo: ModelInfo;
  selectedNeuron: Neuron | null;
  setSelectedNeuron: (neuron: Neuron | null) => void;
  hoveredConnection: { source: string; target: string } | null;
  setHoveredConnection: (conn: { source: string; target: string } | null) => void;
  hoveredNeuron: Neuron | null;
  setHoveredNeuron: (neuron: Neuron | null) => void;
  svgRef: RefObject<SVGSVGElement | null>;
}

export function VisualizationTab(props: VisualizationTabProps) {
  const { modelInfo, selectedNeuron, neurons } = props;

  return (
    <Card className="mt-0 rounded-t-none">
      <CardContent>
        <h2 className="text-lg font-bold mb-2">Visualización de Red Neuronal</h2>
        <NetworkGraph {...props} />
        {selectedNeuron && (
          <div className="text-sm mt-4 bg-gray-50 p-2 rounded border">
            <p>
              <strong>Capa {selectedNeuron.layerIndex}</strong>, Índice {selectedNeuron.neuronIndex}
            </p>
            {/* Estadísticas de Input (primer capa) */}
            {selectedNeuron.layerIndex === Math.min(...neurons.map(n => n.layerIndex)) ? (
              <div className="mt-1">
                <i>Capa de entrada, sin pesos de entrada.</i>
              </div>
            ) : (
              modelInfo.weights?.[selectedNeuron.layerIndex] && (
                (() => {
                  // Pesos de entrada: columna de la neurona en la matriz de la capa actual
                  const weightsIn = modelInfo.weights[selectedNeuron.layerIndex].map(
                    row => row[selectedNeuron.neuronIndex]
                  );
                  const stats = getStats(weightsIn);
                  return stats ? (
                    <div className="mt-1">
                      <strong>Pesos de Entrada:</strong> Min: {formatNumber(stats.min)} | 
                      Max: {formatNumber(stats.max)} | Media: {formatNumber(stats.mean)}
                    </div>
                  ) : null;
                })()
              )
            )}
            
            {/* Solo capa de salida */}
            {selectedNeuron.layerIndex === Math.max(...neurons.map(n => n.layerIndex)) && modelInfo.weights?.[selectedNeuron.layerIndex] && (
              (() => {
                  return (
                    <div className="mt-1">
                       <i>Capa de salida, sin pesos de salida.</i>
                    </div>
                  )
              })()
            )}

             {/* Capas intermedias */}
            {selectedNeuron.layerIndex !== Math.min(...neurons.map(n => n.layerIndex)) &&
              selectedNeuron.layerIndex !== Math.max(...neurons.map(n => n.layerIndex)) &&
              modelInfo.weights?.[selectedNeuron.layerIndex] && modelInfo.weights?.[selectedNeuron.layerIndex + 1] && (
                (() => {
                   const weightsOut = modelInfo.weights[selectedNeuron.layerIndex + 1][selectedNeuron.neuronIndex];
                   const statsOut = getStats(weightsOut);
                   return statsOut ? (
                     <div className="mt-1 text-gray-600">
                        <strong>Pesos de Salida:</strong> Min: {formatNumber(statsOut.min)} | 
                        Max: {formatNumber(statsOut.max)} | Media: {formatNumber(statsOut.mean)}
                     </div>
                   ) : null;
                })()
              )}
            
            {modelInfo.biases?.[selectedNeuron.layerIndex]?.[selectedNeuron.neuronIndex] !== undefined && (
              <div className="mt-1">
                 <strong>Sesgo (Bias):</strong> {formatNumber(modelInfo.biases[selectedNeuron.layerIndex][selectedNeuron.neuronIndex])}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
