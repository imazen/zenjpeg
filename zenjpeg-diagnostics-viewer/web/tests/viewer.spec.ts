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
  expect(diag!.width).toBe(64);
  expect(diag!.height).toBe(64);
  // Default is YCbCr 4:4:4 → 3 components, each 8×8 blocks.
  expect(diag!.components).toHaveLength(3);
  expect(diag!.components[0].blockGrid).toEqual([8, 8]);
  expect(diag!.components[1].blockGrid).toEqual([8, 8]);
  expect(diag!.components[2].blockGrid).toEqual([8, 8]);
  expect(diag!.components[0].blocks).toHaveLength(64);
  // Pre-quant DCT non-trivial on the synthetic pattern.
  const block0 = diag!.components[0].blocks[0];
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
    return await page.evaluate(() => {
      const txt = document.querySelector<HTMLElement>("#status")?.textContent ?? "";
      const m = txt.match(/Encoded (\d+) bytes/);
      return m ? parseInt(m[1]!, 10) : 0;
    });
  };
  const lowQ = await sizeAt("20");
  const hiQ = await sizeAt("95");
  expect(lowQ).toBeGreaterThan(0);
  expect(hiQ).toBeGreaterThan(lowQ);
});

test("Reset EQ button restores h_eq/v_eq to 1", async ({ page }) => {
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
