import { Card, CardContent } from "@/components/ui/card";
import { ModelInfo } from "../../types";
import { formatNumber } from "../../utils/modelHelpers";

export function ParamsTab({ modelInfo }: { modelInfo: ModelInfo }) {
  if (!modelInfo.detailed_weights || modelInfo.detailed_weights.length === 0) return null;

  return (
    <Card className="mt-0 border-2 border-green-300 bg-green-50 rounded-t-none">
      <CardContent>
        <h2 className="text-lg font-bold mb-2 text-green-800">⚖️ Estadísticas de Pesos por Capa</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="bg-green-100">
                <th className="p-2 border border-green-200">Capa</th>
                <th className="p-2 border border-green-200">Tensor</th>
                <th className="p-2 border border-green-200">Shape</th>
                <th className="p-2 border border-green-200">Dtype</th>
                <th className="p-2 border border-green-200">Size</th>
                <th className="p-2 border border-green-200">Mínimo</th>
                <th className="p-2 border border-green-200">Máximo</th>
                <th className="p-2 border border-green-200">Media</th>
                <th className="p-2 border border-green-200">Desv. Est.</th>
              </tr>
            </thead>
            <tbody>
              {modelInfo.detailed_weights.map((layerInfo, idx) =>
                layerInfo.tensors.map((tensor, tIdx) => (
                  <tr key={`${idx}-${tIdx}`} className="hover:bg-green-100 bg-white">
                    {tIdx === 0 && (
                      <td className="p-2 border border-green-200 font-bold" rowSpan={layerInfo.tensors.length}>
                        {layerInfo.layer_name}
                      </td>
                    )}
                    <td className="p-2 border border-green-200 font-mono text-[10px] truncate max-w-[150px]" title={tensor.name}>
                      {tensor.name.split('/').pop()}
                    </td>
                    <td className="p-2 border border-green-200 text-center">[{tensor.shape.join(', ')}]</td>
                    <td className="p-2 border border-green-200 text-center">{tensor.dtype}</td>
                    <td className="p-2 border border-green-200 text-right">{tensor.size}</td>
                    <td className="p-2 border border-green-200 text-right">{formatNumber(tensor.min_value, 4)}</td>
                    <td className="p-2 border border-green-200 text-right">{formatNumber(tensor.max_value, 4)}</td>
                    <td className="p-2 border border-green-200 text-right">{formatNumber(tensor.mean_value, 4)}</td>
                    <td className="p-2 border border-green-200 text-right">{formatNumber(tensor.std_value, 4)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
