// src/components/tabs/FileTab.tsx
import { Card, CardContent } from "@/components/ui/card";
import { ModelInfo } from "../../types";

export function FileTab({ modelInfo }: { modelInfo: ModelInfo }) {
  if (!modelInfo.keras_file_structure) return null;
  return (
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
                file.endsWith(".json")
                  ? "bg-blue-100 text-blue-800"
                  : file.endsWith(".h5")
                  ? "bg-green-100 text-green-800"
                  : "bg-gray-100 text-gray-800"
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
                 <li key={key}>
                     {key}: {String(value)}
                  </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
