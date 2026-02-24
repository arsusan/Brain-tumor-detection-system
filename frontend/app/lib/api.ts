// lib/api.ts
import axios from 'axios';

const api = axios.create({
  // This will use the environment variable if it exists, otherwise it falls back to HF
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'https://arsusan-neuroscan.hf.space',
});

export default api;