import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": "/src"
    },
  },
  server: {
    proxy: {
      '^/(load_model|load_data|layers|activations|prune|save_model)': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
}) 