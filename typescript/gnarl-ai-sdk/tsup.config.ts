import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  // ESM only, like `ai` 7 itself (which also requires Node 22): a CommonJS
  // build would `require` an ESM-only peer.
  format: ["esm"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "es2022",
  treeshake: true,
  splitting: false,
  minify: false,
  // Peers: resolved from the application, never bundled.
  external: ["ai", "gnarl-client", "zod"],
});
