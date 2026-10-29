import { expect, test } from "@playwright/test";

/**
 * E2E suite for the zenjpeg encode-diagnostics viewer.
 *
 * The synthetic 64x64 RGB pattern triggers a default encode on page
 * load; tests then drive the controls and check that diagnostics
 * payloads, heatmaps, and the equalizer-style quant editor respond
 * as expected.
 */

test.beforeEach(async ({ page }) => {
  // Capture page errors so they surface in the trace if the encode
  // pipeline blows up during startup.
  page.on("pageerror", (err) => {
    console.log(`[pageerror] ${err.message}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      console.log(`[console.${msg.type()}] ${msg.text()}`);
    }
  });
  // domcontentloaded is enough — the WASM init() is fired afterwards
  // and we poll on body.dataset.encodeState to know when the initial
  // synthetic encode has settled.
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect
    .poll(
      async () => {
        return await page.evaluate(() => document.body.dataset["encodeState"]);
      },
      { timeout: 60_000 },
    )
    .toBe("done");
  await page.locator('[data-testid="auto-encode-toggle"]').uncheck();
});

test("page loads, triptych populated, diagnostics non-empty", async ({ page }) => {
  // Triptych has three canvases sized > 0.
  for (const id of ["source-canvas", "encoded-canvas", "delta-canvas"]) {
    const dims = await page.locator(`[data-testid="${id}"]`).evaluate((el) => {
      const c = el as HTMLCanvasElement;
      return { w: c.width, h: c.height };
    });
    expect(dims.w).toBeGreaterThan(0);
    expect(dims.h).toBeGreaterThan(0);
  }

  // Diagnostics struct is reachable and well-shaped.
  const diag = await page.evaluate(() => window.__zenjpegDiagnostics.getCurrent());
  expect(diag).not.toBeNull();
  if (!diag) throw new Error("diagnostics null");
  expect(diag.width).toBe(64);
  expect(diag.height).toBe(64);
  // Default is YCbCr 4:4:4 → 3 components, each 8×8 blocks.
  expect(diag.components).toHaveLength(3);
  const [c0, c1, c2] = diag.components;
  if (!c0 || !c1 || !c2) throw new Error("expected 3 components");
  expect(c0.blockGrid).toEqual([8, 8]);
  expect(c1.blockGrid).toEqual([8, 8]);
  expect(c2.blockGrid).toEqual([8, 8]);
  expect(c0.blocks).toHaveLength(64);
  // Pre-quant DCT non-trivial on the synthetic pattern.
  const block0 = c0.blocks[0];
  if (!block0) throw new Error("expected block 0");
  const energy = (block0.coefPreQuant as number[]).reduce(
    (a: number, b: number) => a + Math.abs(b),
    0,
  );
  expect(energy).toBeGreaterThan(0);
});

test("AQ heatmap canvas has been drawn (non-zero pixels)", async ({ page }) => {
  const drawn = await page.evaluate(() => {
    const c = document.querySelector<HTMLCanvasElement>(
      '[data-testid="aq-canvas"]',
    )!;
    if (!c.width || !c.height) return false;
    const ctx = c.getContext("2d");
    if (!ctx) return false;
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i] !== 0 || d[i + 1] !== 0 || d[i + 2] !== 0) return true;
    }
    return false;
  });
  expect(drawn).toBe(true);
});

test("dragging an h_eq slider re-renders the utilization heatmap", async ({
  page,
}) => {
  await page.locator('[data-testid="tab-quant"]').click();
  // Snapshot the utilization canvas before the change.
  const beforeHash = await canvasFingerprint(page, '[data-testid="comp-0-utilization"]');

  // Move h_eq[0] hard. With the default starts at 1 and step 0.05.
  const slider = page.locator('[data-testid="comp-0-h-eq-0"]');
  await slider.evaluate((el) => {
    const i = el as HTMLInputElement;
    i.value = "2";
    i.dispatchEvent(new Event("input", { bubbles: true }));
  });
  // Wait briefly for the redraw.
  await page.waitForTimeout(50);
  const afterHash = await canvasFingerprint(page, '[data-testid="comp-0-utilization"]');
  expect(afterHash).not.toBe(beforeHash);
});

test("changing color path to XYB re-encodes and updates components", async ({
  page,
}) => {
  await page.locator('[data-testid="color-path-select"]').selectOption("xyb");
  // Subsampling label hides; XYB B label shows.
  await expect(page.locator("#subsampling-label")).toBeHidden();
  await expect(page.locator("#xyb-subsampling-label")).toBeVisible();
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(async () => {
    return await page.evaluate(() => {
      const d = window.__zenjpegDiagnostics.getCurrent();
      return d?.colorPath;
    });
  }, { timeout: 60_000 }).toBe("XYB");
  await page.locator('[data-testid="xyb-subsampling-select"]').selectOption("bQuarter");
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(async () => page.evaluate(() => window.__zenjpegDiagnostics.getCurrent()?.samplingFactors)).toEqual([[2, 2], [2, 2], [1, 1]]);
});

test("subsampling 4:2:0 produces half-resolution chroma block grids", async ({
  page,
}) => {
  await page.locator('[data-testid="subsampling-select"]').selectOption("quarter");
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(async () => {
    return await page.evaluate(() => {
      const d = window.__zenjpegDiagnostics.getCurrent();
      if (!d) return null;
      return [
        d.components[0]?.blockGrid,
        d.components[1]?.blockGrid,
        d.components[2]?.blockGrid,
      ];
    });
  }, { timeout: 60_000 }).toEqual([
    [8, 8],
    [4, 4],
    [4, 4],
  ]);
});

test("quality slider drag changes encoded byte size", async ({ page }) => {
  const sizeAt = async (q: string) => {
    await page.locator('[data-testid="quality-slider"]').evaluate(
      (el, value) => {
        const i = el as HTMLInputElement;
        i.value = value;
        i.dispatchEvent(new Event("input", { bubbles: true }));
      },
      q,
    );
    await page.locator('[data-testid="encode-button"]').click();
    await expect.poll(async () => {
      return await page.evaluate(() => document.body.dataset["encodeState"]);
    }, { timeout: 60_000 }).toBe("done");
    return await page.evaluate(() => window.__zenjpegDiagnostics.getLastEncode()?.bytes ?? 0);
  };
  const lowQ = await sizeAt("20");
  const hiQ = await sizeAt("95");
  expect(lowQ).toBeGreaterThan(0);
  expect(hiQ).toBeGreaterThan(lowQ);
});

test("Reset EQ button restores h_eq/v_eq to 1", async ({ page }) => {
  await page.locator('[data-testid="tab-quant"]').click();
  await page.locator('[data-testid="comp-0-h-eq-3"]').evaluate((el) => {
    const i = el as HTMLInputElement;
    i.value = "1.7";
    i.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.locator('[data-testid="comp-0-reset-eq"]').click();
  const value = await page
    .locator('[data-testid="comp-0-h-eq-3"]')
    .evaluate((el) => (el as HTMLInputElement).value);
  expect(parseFloat(value)).toBeCloseTo(1.0, 2);
});

async function canvasFingerprint(page: import("@playwright/test").Page, sel: string): Promise<string> {
  return page.evaluate((s) => {
    const c = document.querySelector<HTMLCanvasElement>(s);
    if (!c) return "";
    const ctx = c.getContext("2d");
    if (!ctx) return "";
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    // Cheap rolling hash over the data — exact pixel equality is not
    // important, we just need a stable fingerprint that changes when
    // the canvas content changes.
    let h = 5381;
    for (let i = 0; i < d.length; i += 64) {
      h = ((h << 5) + h + d[i]!) | 0;
    }
    return String(h);
  }, sel);
}


test("table edit is encoded and survives a quality change", async ({ page }) => {
  await page.locator('[data-testid="tab-quant"]').click();
  const slider = page.locator('[data-testid="comp-0-h-eq-0"]');
  await slider.evaluate((el) => {
    const input = el as HTMLInputElement;
    input.value = "2";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  await expect(page.locator('[data-testid="status"]')).toContainText("custom Q");
  const edited = await page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().customQuantTables);
  expect(edited?.y).toHaveLength(64);
  expect(edited?.cb).toHaveLength(64);
  expect(edited?.cr).toHaveLength(64);
  const actual = await page.evaluate(() => window.__zenjpegDiagnostics.getCurrent()?.components[0]?.quantTableBase[0]);
  expect(actual).toBe(Math.round(edited!.y[0]!));
  await page.locator('[data-testid="tab-quality"]').click();
  await page.locator('[data-testid="quality-slider"]').evaluate((el) => {
    const input = el as HTMLInputElement;
    input.value = "65";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  await expect(page.locator('[data-testid="status"]')).toContainText("custom Q");
  expect(await slider.inputValue()).toBe("2");
});

test("upload and zoom inspector respond to interaction", async ({ page }) => {
  const png = await page.evaluate(() => {
    const c = document.createElement("canvas");
    c.width = 64; c.height = 48;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#2596be"; ctx.fillRect(0, 0, 64, 48);
    ctx.fillStyle = "#a53a62"; ctx.fillRect(3, 7, 20, 30);
    ctx.clearRect(0, 0, 2, 2);
    return c.toDataURL("image/png").split(",")[1]!;
  });
  await page.locator('[data-testid="image-input"]').setInputFiles({ name: "test.png", mimeType: "image/png", buffer: Buffer.from(png, "base64") });
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getCurrent()?.height)).toBe(48);
  const alpha = await page.locator('[data-testid="source-canvas"]').evaluate((el) => (el as HTMLCanvasElement).getContext("2d")!.getImageData(0, 0, 1, 1).data[3]);
  expect(alpha).toBe(255);
  await page.locator('[data-testid="source-canvas"]').click();
  await expect(page.locator('[data-testid="zoom-modal"]')).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.locator('[data-testid="zoom-modal"]')).toBeHidden();
});
