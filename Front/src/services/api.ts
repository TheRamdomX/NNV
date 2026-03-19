import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export const loadModelApi = async (modelFile: File) => {
  const form = new FormData();
  form.append('file', modelFile);
  return await api.post('/load_model', form);
};

export const uploadNpyApi = async (npyFile: File) => {
  const form = new FormData();
  form.append('npy', npyFile);
  return await api.post('/carga_parametros', form);
};

export const fetchModelInfoApi = async () => {
  const res = await api.get('/model_info');
  return res.data;
};

export default api;
