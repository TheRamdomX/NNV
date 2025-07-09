import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from '@/components/ui/select';
import axios from 'axios';
import ReactFlow, { Background, Controls, Node, Edge } from 'reactflow';
import 'reactflow/dist/style.css';

interface Layer {
  index: number;
  name: string;
  output_shape?: number[];
  activation?: string | null;
}

interface Activation {
  neuron_index: number;
  mean_activation: number;
  status: string;
}

interface PruneResult {
  layer_index: number;
  pruned_neurons: number;
  total_neurons: number;
}

export default function App() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [layers, setLayers] = useState<Layer[]>([]);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [thresholdLow, setThresholdLow] = useState(0.01);
  const [thresholdHigh, setThresholdHigh] = useState(0.99);
  const [activations, setActivations] = useState<Activation[]>([]);
  const [pruneResults, setPruneResults] = useState<PruneResult[]>([]);
  const [fileName, setFileName] = useState('model_pruned');

  const loadModel = async () => {
    if (!modelFile) return;
    const form = new FormData(); form.append('file', modelFile);
    await axios.post('/load_model', form);
    alert('Modelo cargado');
    fetchLayers();
  };

  const loadData = async () => {
    if (!dataFile) return;
    const form = new FormData(); form.append('file', dataFile);
    const res = await axios.post('/load_data', form);
    alert(`Datos cargados: shape ${res.data.shape}`);
  };

  const fetchLayers = async () => {
    const res = await axios.get('/layers');
    setLayers(res.data);
  };

  const analyze = async () => {
    if (selectedLayer === null) {
      alert("Selecciona una capa primero");
      return;
    }
    const res = await axios.post('/activations', {
      layer_index: selectedLayer,
      threshold_low: thresholdLow,
      threshold_high: thresholdHigh,
    });
    setActivations(res.data);
  };

  const prune = async () => {
    const res = await axios.post('/prune', [
      { layer_index: selectedLayer, threshold_low: thresholdLow }
    ]);
    setPruneResults(res.data);
    alert('Poda completada');
  };

  const saveModel = async () => {
    const res = await axios.post('/save_model', null, { params: { file_name: fileName } });
    alert(res.data.message);
  };

  // --- Grafo de capas ---
  const nodes: Node[] = layers.map((layer, i) => ({
    id: String(layer.index),
    data: { label: `${layer.name}\n${layer.output_shape ? layer.output_shape.join('x') : ''}` },
    position: { x: 100 * i, y: 0 },
  }));
  const edges: Edge[] = layers.slice(1).map((layer, i) => ({
    id: `e${i}-${i+1}`,
    source: String(layers[i].index),
    target: String(layer.index),
    type: 'smoothstep',
  }));

  // --- Tabla de neuronas por capa densa ---
  const denseLayers = layers.filter(l => l.output_shape && l.output_shape.length > 0 && l.output_shape[l.output_shape.length - 1] > 1);

  // --- Grafo de activaciones ---
  let activationNodes: Node[] = [];
  let activationEdges: Edge[] = [];
  if (activations.length > 0 && selectedLayer !== null) {
    activationNodes = activations.map((a, i) => ({
      id: String(a.neuron_index),
      data: { label: `N${a.neuron_index}\n${a.status}` },
      position: { x: 80 * (i % 10), y: 80 * Math.floor(i / 10) },
      style: {
        background: a.status === 'dead' ? '#f87171' : a.status === 'active' ? '#34d399' : '#fbbf24',
        color: '#222',
        border: '1px solid #888',
        borderRadius: 8,
        width: 60,
        height: 40,
        fontSize: 12,
        textAlign: 'center',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        whiteSpace: 'pre-line',
      },
    }));
    // Opcional: conectar neuronas en secuencia
    activationEdges = activations.slice(1).map((a, i) => ({
      id: `an${i}-${i+1}`,
      source: String(activations[i].neuron_index),
      target: String(a.neuron_index),
      type: 'smoothstep',
    }));
  }

  return (
    <div className="grid">
      <Card>
        <CardContent>
          <h2 className="text-xl font-bold mb-2">Carga de Modelo y Datos</h2>
          <Input type="file" onChange={e => setModelFile(e.target.files?.[0] || null)} />
          <Button onClick={loadModel} className="mt-2">Cargar Modelo</Button>
          <Input type="file" onChange={e => setDataFile(e.target.files?.[0] || null)} className="mt-4" />
          <Button onClick={loadData} className="mt-2">Cargar Datos</Button>
        </CardContent>
      </Card>

      {layers.length > 0 && (
        <Card>
          <CardContent>
            <h2 className="text-xl font-bold mb-2">Grafo del Modelo</h2>
            <div style={{ width: '100%', height: 300, background: '#f9f9f9', borderRadius: 8 }}>
              <ReactFlow nodes={nodes} edges={edges} fitView>
                <Background />
                <Controls />
              </ReactFlow>
            </div>
          </CardContent>
        </Card>
      )}

      {denseLayers.length > 0 && (
        <Card>
          <CardContent>
            <h2 className="text-xl font-bold mb-2">Neurona a Neurona (por capa densa)</h2>
            {denseLayers.map(layer => (
              <div key={layer.index} className="mb-4">
                <h3 className="font-bold mb-1">{layer.name} ({layer.output_shape?.join('x')})</h3>
                <div className="overflow-auto max-h-64">
                  <table className="w-full table-auto">
                    <thead>
                      <tr><th>Neurona</th></tr>
                    </thead>
                    <tbody>
                      {Array.from({ length: layer.output_shape ? layer.output_shape[layer.output_shape.length - 1] : 0 }).map((_, i) => (
                        <tr key={i}>
                          <td>{i}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent>
          <h2 className="text-xl font-bold mb-2">Análisis de Activaciones</h2>
          <Select onValueChange={value => setSelectedLayer(Number(value))}>
            <SelectTrigger>
              <SelectValue placeholder="Selecciona capa" />
            </SelectTrigger>
            <SelectContent>
              {layers.map(l => (
                <SelectItem key={l.index} value={String(l.index)}>
                  {l.index} - {l.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <Input type="number" value={thresholdLow} step="0.01" onChange={e => setThresholdLow(parseFloat(e.target.value))} placeholder="Umbral bajo" />
            <Input type="number" value={thresholdHigh} step="0.01" onChange={e => setThresholdHigh(parseFloat(e.target.value))} placeholder="Umbral alto" />
          </div>
          <div className="flex mt-2">
            <Button onClick={analyze}>Analizar</Button>
            <Button onClick={prune} variant="secondary">Podar</Button>
          </div>
        </CardContent>
      </Card>

      {activations.length > 0 && (
        <Card>
          <CardContent>
            <h2 className="text-xl font-bold mb-2">Resultados de Activaciones</h2>
            <div className="overflow-auto max-h-64">
              <table className="w-full table-auto">
                <thead>
                  <tr><th>Índice</th><th>Media</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {activations.map(a => (
                    <tr key={a.neuron_index} className={a.status === 'dead' ? 'opacity-50' : ''}>
                      <td>{a.neuron_index}</td>
                      <td>{a.mean_activation.toFixed(4)}</td>
                      <td>{a.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {pruneResults.length > 0 && (
        <Card>
          <CardContent>
            <h2 className="text-xl font-bold mb-2">Resultados de Poda</h2>
            <ul className="list-disc list-inside">
              {pruneResults.map(r => (
                <li key={r.layer_index}>
                  Capa {r.layer_index}: {r.pruned_neurons} de {r.total_neurons} neuronas podadas.
                </li>
              ))}
            </ul>
            <div className="flex mt-2">
              <Input placeholder="Nombre archivo" value={fileName} onChange={e => setFileName(e.target.value)} />
              <Button onClick={saveModel}>Guardar Modelo</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Grafo de activaciones al final de la página */}
      {activationNodes.length > 0 && (
        <Card>
          <CardContent>
            <h2 className="text-xl font-bold mb-2">Grafo de Activaciones (neurona a neurona)</h2>
            <div style={{ width: '100%', height: 400, background: '#f9f9f9', borderRadius: 8 }}>
              <ReactFlow nodes={activationNodes} edges={activationEdges} fitView>
                <Background />
                <Controls />
              </ReactFlow>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
