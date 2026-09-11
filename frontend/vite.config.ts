import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// バックエンド（FastAPI）は 8000 番で動く前提。
// 開発時は Vite が /ws と /api を 8000 番へプロキシするので、
// フロントのコードは同一オリジンの相対パスだけを見ればよい。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
      },
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    // three と drei は大きいので、警告のしきい値を上げておく
    chunkSizeWarningLimit: 2000,
  },
})
