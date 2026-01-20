import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 5001,
        host: '0.0.0.0',
        proxy: {
          '/ws': {
          target: 'ws://localhost:5555',
          ws: true,
          changeOrigin: true
        },
        '/video_in': {
          target: 'ws://localhost:5555',
          ws: true,
          changeOrigin: true
        }
        },
        fs: {
          allow: [path.resolve(__dirname, '..'), path.resolve(__dirname, '../checkpoints')]
        }
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
