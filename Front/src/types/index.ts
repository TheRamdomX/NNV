export interface Layer {
  index: number;
  name: string;
  output_shape?: number[];
  input_shape?: number[];
  activation?: string | null;
  type?: string;
  config?: any;
  neuron_count?: number;
  trainable?: boolean;
  dtype?: string;
}

export interface Neuron {
  id: string;
  x: number;
  y: number;
  layerIndex: number;
  neuronIndex: number;
}

export interface Connection {
  source: string;
  target: string;
}

export interface ModelMetadata {
  model_name: string;
  created_at: string;
  trainable_params: number;
  non_trainable_params: number;
  total_params: number;
  keras_version?: string;
  backend?: string;
  date_saved?: string;
}

export interface OptimizerInfo {
  type: string;
  config: any;
}

export interface KerasFileStructure {
  files: string[];
  config_json?: any;
  metadata_json?: any;
}

export interface WeightTensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  size: number;
  min_value: number;
  max_value: number;
  mean_value: number;
  std_value: number;
}

export interface LayerWeightsInfo {
  layer_name: string;
  tensors: WeightTensorInfo[];
}

export interface ModelInfo {
  summary: string;
  layers: Layer[];
  optimizer: OptimizerInfo;
  loss: string;
  metrics: string[];
  metadata: ModelMetadata;
  weights: number[][][];
  biases: number[][];
  keras_file_structure?: KerasFileStructure;
  detailed_weights?: LayerWeightsInfo[];
}

export type TabType = 'file' | 'model' | 'params' | 'visualization';
