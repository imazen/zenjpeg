import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://localhost:3173",
    trace: "retain-on-failure",
  },
  // Per-test timeout: WASM compile + first encode in headless chromium
  // can take 20+ seconds in WSL2 / GitHub Actions runners. 90s gives
  // headroom; the suite still completes in well under five minutes.
  timeout: 90_000,
  expect: {
    timeout: 45_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    // Build then preview — gives us the wasm-bundled output served as
    // ESM, exactly like a real deployment. Preview is faster than dev
    // for CI and avoids HMR noise. We reuse any existing dev preview
    // (so a manually-started `vite preview --port 3173` is fine);
    // outside CI this avoids the build cost on every test invocation.
    command: "npx vite build && npx vite preview --port 3173 --strictPort",
    url: "http://localhost:3173",
    reuseExistingServer: true,
    timeout: 180_000,
  },
});
