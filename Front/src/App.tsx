// NNV - Neural Network Visualizer
// Aplicación para visualizar y analizar modelos de redes neuronales Keras

import { useEffect } from 'react';
import { useModelData } from './hooks/useModelData';
import { ModelUploader } from './components/ModelUploader';
import { TabNavigation } from './components/tabs/TabNavigation';
import { FileTab } from './components/tabs/FileTab';
import { ModelTab } from './components/tabs/ModelTab';
import { ParamsTab } from './components/tabs/ParamsTab';
import { VisualizationTab } from './components/tabs/VisualizationTab';

export default function App() {
  const {
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
    buildNetwork
  } = useModelData();

  useEffect(() => {
    if (layers.length > 0 && activeTab === 'visualization') {
      setTimeout(() => {
        if (svgRef.current && svgRef.current.clientWidth > 0 && svgRef.current.clientHeight > 0) {
          buildNetwork(layers);
        } else {
          setTimeout(() => {
            if (svgRef.current && svgRef.current.clientWidth > 0) {
              buildNetwork(layers);
            }
          }, 200);
        }
      }, 100);
    }
  }, [layers, activeTab, buildNetwork, svgRef]);

  return (
    <div className="p-4">
      <ModelUploader
        setModelFile={setModelFile}
        setNpyFile={setNpyFile}
        loadModel={loadModel}
        uploadNpy={uploadNpy}
        modelFile={modelFile}
        npyFile={npyFile}
      />

      {modelInfo && (
        <>
          <TabNavigation activeTab={activeTab} setActiveTab={setActiveTab} />
          
          {activeTab === 'file' && <FileTab modelInfo={modelInfo} />}
          
          {activeTab === 'model' && (
            <ModelTab 
              modelInfo={modelInfo} 
              selectedLayerIdx={selectedLayerIdx} 
              setSelectedLayerIdx={setSelectedLayerIdx} 
            />
          )}

          {activeTab === 'params' && <ParamsTab modelInfo={modelInfo} />}

          {activeTab === 'visualization' && (
            <VisualizationTab
              layers={layers}
              neurons={neurons}
              connections={connections}
              modelInfo={modelInfo}
              selectedNeuron={selectedNeuron}
              setSelectedNeuron={setSelectedNeuron}
              hoveredConnection={hoveredConnection}
              setHoveredConnection={setHoveredConnection}
              hoveredNeuron={hoveredNeuron}
              setHoveredNeuron={setHoveredNeuron}
              svgRef={svgRef}
            />
          )}
        </>
      )}
    </div>
  );
}