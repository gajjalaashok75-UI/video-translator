/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI process so the browser only ever
// talks to one origin. That is the same shape the app will have in production
// (FastAPI serving the built bundle), so no component ever needs an absolute
// backend URL and nothing changes when the two are co-located.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Vite blocks requests whose Host header it does not recognise. That is a
    // sensible default, so it is left on unless a deployment sets
    // VITE_ALLOWED_HOSTS (comma-separated) to reach the dev server through a
    // tunnel or proxy. No hostname is committed here.
    allowedHosts: process.env.VITE_ALLOWED_HOSTS
      ? process.env.VITE_ALLOWED_HOSTS.split(",").map((host) => host.trim())
      : undefined,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    css: false,
  },
});
