// src/hooks/useModelData.ts
import { useState, useCallback, useRef } from 'react';
import { loadModelApi, uploadNpyApi, fetchModelInfoApi } from '../services/api';
import { ModelInfo, Layer, Neuron, Connection, TabType } from '../types';

export function useModelData() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [npyFile, setNpyFile] = useState<File | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [layers, setLayers] = useState<Layer[]>([]);
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedNeuron, setSelectedNeuron] = useState<Neuron | null>(null);
  const [hoveredConnection, setHoveredConnection] = useState<{ source: string, target: string } | null>(null);
  const [hoveredNeuron, setHoveredNeuron] = useState<Neuron | null>(null);
  const [selectedLayerIdx, setSelectedLayerIdx] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('file');

  const svgRef = useRef<SVGSVGElement>(null);

  const fetchModelInfo = useCallback(async () => {
    try {
      const info = await fetchModelInfoApi();
      setModelInfo(info);
      setLayers(info.layers);
    } catch (error) {
      console.error("Failed to fetch model info", error);
    }
  }, []);

  const loadModel = useCallback(async () => {
    if (!modelFile) return;
    try {
      await loadModelApi(modelFile);
      await fetchModelInfo();
    } catch (error) {
       console.error("Failed to load model", error);
    }
  }, [modelFile, fetchModelInfo]);

  const uploadNpy = useCallback(async () => {
    if (!npyFile) return;
    try {
      await uploadNpyApi(npyFile);
    } catch (error) {
      console.error("Failed to upload npy", error);
    }
  }, [npyFile]);


  const buildNetwork = useCallback((layerData: Layer[]) => {
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
  }, []);

  return {
    modelFile,
    setModelFile,
    npyFile,
    setNpyFile,
    modelInfo,
    layers,
    neurons,
    connections,
    selectedNeuron,
    setSelectedNeuron,
    hoveredConnection,
    setHoveredConnection,
    hoveredNeuron,
    setHoveredNeuron,
    selectedLayerIdx,
    setSelectedLayerIdx,
    activeTab,
    setActiveTab,
    svgRef,
    loadModel,
    uploadNpy,
    fetchModelInfo,
    buildNetwork
  };
}
