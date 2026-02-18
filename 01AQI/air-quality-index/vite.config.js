import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'


export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/aqi-api': {
        target: 'https://hub.juheapi.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/aqi-api/, ''),
      },
    },
  },
})