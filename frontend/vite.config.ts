/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const BACKEND = "http://127.0.0.1:7860";

// The SPA is served by the Python backend from autotune_studio/web, possibly under a
// non-root proxy path (Colab / Jupyter server-proxy), so every asset URL is relative.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  build: {
    // NOTE: emptyOutDir wipes ../autotune_studio/web, including its .gitkeep placeholder.
    outDir: "../autotune_studio/web",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: false },
      "/ws": { target: BACKEND, ws: true, changeOrigin: false },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    restoreMocks: true,
  },
});
