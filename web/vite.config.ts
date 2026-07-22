import { defineConfig } from "vite";
import { configDefaults } from "vitest/config";

const crossOriginIsolationHeaders = {
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "same-origin",
} as const;

export default defineConfig({
  build: {
    target: "esnext",
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
  server: {
    headers: crossOriginIsolationHeaders,
    port: 5173,
    strictPort: true,
    // Query-gated correctness diagnostics read the checked-in native oracle
    // from ../.research through Vite's /@fs/ route. Production builds do not
    // expose this development-only path.
    fs: { allow: [".."] },
  },
  test: {
    environment: "node",
    // These use Node's built-in test runner so they can import the benchmark
    // CLI as ordinary ESM without Vitest trying to reinterpret the files.
    exclude: [
      ...configDefaults.exclude,
      "scripts/benchmark/**/*.test.mjs",
    ],
  },
  worker: {
    format: "es",
  },
});
