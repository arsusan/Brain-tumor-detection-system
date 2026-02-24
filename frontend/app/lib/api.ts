import axios from 'axios';

const api = axios.create({
  baseURL: 'https://arsusan-neuroscan.hf.space', // Using 127.0.0.1 is more stable for local dev
});

export default api;