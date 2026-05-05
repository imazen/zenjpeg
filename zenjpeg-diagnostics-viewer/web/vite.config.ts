import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset URLs (`./assets/...`) so the same `dist/` works at
  // any deploy subpath: local `vite preview` at root, GitHub Pages at
  // `/zenjpeg/diagnostics/`, Cloudflare Pages at `/`, an arbitrary
  // pre-existing CDN, etc. wasm-bindgen's `new URL("./assets/...", import.meta.url)`
  // resolves correctly from index.html either way.
  base: "./",
  server: {
    // We're prohibited from binding port 8080 (reserved on the host).
    // Pick a port in the 3000-3999 range; 3173 is the diagnostics
    // viewer's default.
    port: 3173,
    strictPort: true,
    fs: {
      // Allow vite to serve files from the wasm-pkg sibling directory.
      allow: [".."],
    },
  },
  preview: {
    port: 3173,
    strictPort: true,
  },
  build: {
    target: "esnext",
  },
  optimizeDeps: {
    // wasm-pack output uses ES modules — let vite serve them as-is.
    exclude: ["./wasm-pkg/zenjpeg_diagnostics_wasm.js"],
  },
});
