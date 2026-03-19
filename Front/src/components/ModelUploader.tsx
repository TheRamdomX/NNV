// src/components/ModelUploader.tsx
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface ModelUploaderProps {
  setModelFile: (file: File | null) => void;
  setNpyFile: (file: File | null) => void;
  loadModel: () => void;
  uploadNpy: () => void;
  modelFile: File | null;
  npyFile: File | null;
}

export function ModelUploader({
  setModelFile,
  setNpyFile,
  loadModel,
  uploadNpy,
  modelFile,
  npyFile,
}: ModelUploaderProps) {
  return (
    <Card>
      <CardContent>
        <h2 className="text-lg font-bold mb-2">Cargar modelo</h2>
        <div className="flex flex-col gap-2">
          <Input
            type="file"
            accept=".keras"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file && file.name.endsWith('.keras')) setModelFile(file);
              else setModelFile(null);
            }}
          />
          <Button onClick={loadModel} className="mt-2" disabled={!modelFile}>
            Cargar modelo
          </Button>
          <Input
            type="file"
            accept=".npy"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file && file.name.endsWith('.npy')) setNpyFile(file);
              else setNpyFile(null);
            }}
          />
          <Button onClick={uploadNpy} className="mt-2" disabled={!npyFile}>
            Cargar parámetros
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
