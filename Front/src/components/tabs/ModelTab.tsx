// src/components/tabs/ModelTab.tsx
import { Card, CardContent } from "@/components/ui/card";
import { ModelInfo } from "../../types";
import { renderConfig } from "../../utils/modelHelpers";
import { MiniNetworkPreview } from "../visualization/MiniNetworkPreview";

interface ModelTabProps {
  modelInfo: ModelInfo;
  selectedLayerIdx: number | null;
  setSelectedLayerIdx: (idx: number | null) => void;
}

export function ModelTab({ modelInfo, selectedLayerIdx, setSelectedLayerIdx }: ModelTabProps) {
  return (
    <>
      <Card className="mt-0 rounded-t-none">
        <CardContent>
          <h2 className="text-lg font-bold mb-2">Detalles del Modelo</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm mb-2">
                Nombre: {modelInfo.metadata.model_name}
              </p>
              <p className="text-sm mb-2">
                Parámetros: {modelInfo.metadata.total_params}
              </p>
              <p className="text-sm mb-2">
                Función de pérdida: {modelInfo.loss}
              </p>
              <p className="text-sm mb-2">
                Optimizador: {modelInfo.optimizer.type}
              </p>
              <p className="text-sm mb-4">
                Métricas: {modelInfo.metrics.join(", ")}
              </p>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <p className="text-sm font-semibold mb-2">Información del archivo .keras:</p>
              {modelInfo.metadata.keras_version && (
                <p className="text-xs">Keras version: {modelInfo.metadata.keras_version}</p>
              )}
              {modelInfo.metadata.backend && (
                <p className="text-xs">Backend: {modelInfo.metadata.backend}</p>
              )}
              {modelInfo.metadata.date_saved && (
                <p className="text-xs">Date saved: {modelInfo.metadata.date_saved}</p>
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
                    : layer.type === 'Dense'
                    ? 'bg-blue-50 border-blue-300 text-blue-700 hover:bg-blue-100'
                    : layer.type === 'InputLayer'
                    ? 'bg-green-50 border-green-300 text-green-700 hover:bg-green-100'
                    : layer.type === 'Dropout'
                    ? 'bg-yellow-50 border-yellow-300 text-yellow-700 hover:bg-yellow-100'
                    : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
                onClick={() => setSelectedLayerIdx(selectedLayerIdx === idx ? null : idx)}
              >
                {layer.name}
              </button>
            ))}
          </div>

          {/* Grid con detalles y mini previsualización */}
          {selectedLayerIdx !== null && modelInfo.layers[selectedLayerIdx] && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Detalles de la capa (2/3) */}
              <Card className="lg:col-span-2 border-4 rounded-lg border-blue-600 bg-blue-50">
                <CardContent>
                  <h3 className="text-lg font-bold mt-2">Capa: {modelInfo.layers[selectedLayerIdx].name}</h3>
                  <div className="text-sm">
                    <p><strong>Tipo:</strong> {modelInfo.layers[selectedLayerIdx].type}</p>
                    <p><strong>Índice:</strong> {modelInfo.layers[selectedLayerIdx].index}</p>
                    {modelInfo.layers[selectedLayerIdx].neuron_count !== undefined && (
                      <p><strong>Neuronas:</strong> {modelInfo.layers[selectedLayerIdx].neuron_count}</p>
                    )}
                    {modelInfo.layers[selectedLayerIdx].activation && (
                      <p><strong>Activación:</strong> {modelInfo.layers[selectedLayerIdx].activation}</p>
                    )}
                    <div className="mt-2">
                      <strong>Configuración:</strong>
                      {renderConfig(modelInfo.layers[selectedLayerIdx].config)}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Mini previsualización (1/3) */}
              <Card className="border-2 rounded-lg border-orange-400 bg-orange-50 flex items-center justify-center p-4">
                <MiniNetworkPreview layers={modelInfo.layers} highlightLayerIdx={selectedLayerIdx} />
              </Card>
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}
