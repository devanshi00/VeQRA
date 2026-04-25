import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  server: {
    host: true,
    allowedHosts: true, 
    proxy: {
      "/api": {
        target: "http://localhost:5000", // your backend
        changeOrigin: true,
        secure: false,
      },
    },
  },
  plugins: [react(),tailwindcss(),],
})
