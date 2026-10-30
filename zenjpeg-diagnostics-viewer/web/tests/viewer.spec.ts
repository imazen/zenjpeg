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
  await page.locator('[data-testid="controls-toggle"]').click();
  await expect(page.locator('[data-testid="controls-popover"]')).toBeVisible();
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
  await page.locator('label:has(input[name="color-choice"][value="xyb"])').click();
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
  await page.locator('label:has(input[name="xyb-sub-choice"][value="bQuarter"])').click();
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(async () => page.evaluate(() => window.__zenjpegDiagnostics.getCurrent()?.samplingFactors)).toEqual([[2, 2], [2, 2], [1, 1]]);
});

test("subsampling 4:2:0 produces half-resolution chroma block grids", async ({
  page,
}) => {
  await page.locator('label:has(input[name="sub-choice"][value="quarter"])').click();
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
  await page.locator('[data-testid="tab-aq"]').click();
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


test("settings and cells can be edited without a browser prompt", async ({ page }) => {
  await page.locator('[data-testid="tab-quant"]').click();
  await page.locator('[data-testid="component-pick-1"]').click();
  await expect(page.locator('[data-component="1"]')).toBeVisible();
  await expect(page.locator('[data-component="0"]')).toBeHidden();
  const cell = page.locator('[data-testid="comp-1-cell-0"]');
  await cell.click();
  const editor = page.locator('[data-testid="cell-editor"]');
  await expect(editor).toBeVisible();
  const input = page.locator('[data-testid="cell-editor-value"]');
  await expect(input).toBeFocused();
  await input.fill("42");
  await page.locator('[data-testid="cell-editor-apply"]').click();
  await expect(editor).toBeHidden();
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().customQuantTables?.cb[0])).toBe(42);
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getCurrent()?.components[1]?.quantTableBase[0])).toBe(42);
  await expect(page.locator('[data-component="1"]')).toBeVisible();
});

test("settings support keyboard navigation and escape", async ({ page }) => {
  const modeTab = page.locator('[data-testid="tab-aq"]');
  await modeTab.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator('[data-testid="tab-quant"]')).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("Escape");
  await expect(page.locator('[data-testid="controls-popover"]')).toBeHidden();
  await expect(page.locator('[data-testid="controls-toggle"]')).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator('[data-testid="controls-popover"]')).toBeVisible();
});

test("Huffman controls show the effective strategy and preserve sequential preference", async ({ page }) => {
  await page.locator('.settings-details summary').filter({ hasText: "More encoder settings" }).click();
  const huffman = page.locator('[data-testid="optimize-huffman-checkbox"]');
  const progressive = page.locator('[data-testid="progressive-checkbox"]');
  const note = page.locator('[data-testid="huffman-note"]');
  await expect(progressive).toBeChecked();
  await expect(huffman).toBeChecked();
  await expect(huffman).toBeDisabled();
  await expect(note).toContainText("Progressive scans always use optimized");
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().optimizeHuffman)).toBe(true);
  await page.locator('label:has(input[name="scan"][value="sequential"])').click();
  await expect(huffman).toBeEnabled();
  await huffman.uncheck();
  const encode = async () => {
    await page.evaluate(() => { document.body.dataset["encodeState"] = "waiting"; });
    await page.locator('[data-testid="encode-button"]').click();
    await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
    return page.evaluate(() => ({
      size: window.__zenjpegDiagnostics.getLastEncode()!.bytes,
      pixels: Array.from(document.querySelector<HTMLCanvasElement>('[data-testid="encoded-canvas"]')!.getContext("2d")!.getImageData(0, 0, 64, 64).data),
    }));
  };
  const fixed = await encode();
  await huffman.check();
  const optimized = await encode();
  expect(optimized.size).not.toBe(fixed.size);
  expect(optimized.pixels).toEqual(fixed.pixels);
  await huffman.uncheck();
  await page.locator('label:has(input[name="scan"][value="progressive"])').click();
  await expect(huffman).toBeChecked();
  await expect(huffman).toBeDisabled();
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().optimizeHuffman)).toBe(true);
  await page.locator('label:has(input[name="scan"][value="sequential"])').click();
  await expect(huffman).toBeEnabled();
  await expect(huffman).not.toBeChecked();
});

test("mode and color decisions reveal the settings that apply", async ({ page }) => {
  await expect(page.locator('[data-testid="tab-aq"]')).toHaveAttribute("aria-selected", "true");
  await expect(page.locator('[data-testid="mode-baseline"]')).toBeChecked();
  await expect(page.locator('[data-testid="aq-checkbox"]')).toBeChecked();
  await expect(page.locator('[data-testid="color-resolved"]')).toContainText("YCbCr");
  await expect(page.locator('[data-testid="sampling-resolved"]')).toContainText("4:4:4");
  await expect(page.locator('[data-testid="scan-resolved"]')).toContainText("progressive");
  await expect(page.locator('#trellis-settings')).toBeHidden();
  await expect(page.locator('#hybrid-settings')).toBeHidden();
  await page.locator('label:has(input[name="sub-choice"][value="quarter"])').click();
  await page.locator('.settings-details summary').filter({hasText:"More encoder settings"}).click();
  await expect(page.locator('[data-testid="sharp-yuv-checkbox"]')).toBeVisible();
  await page.locator('label:has([data-testid="mode-trellis"])').click();
  await expect(page.locator('#trellis-settings')).toBeVisible();
  await expect(page.locator('#hybrid-settings')).toBeHidden();
  await page.locator('#trellis-settings summary').click();
  await expect(page.locator('[data-testid="trellis-delta-dc"]')).toBeHidden();
  await page.locator('[data-testid="trellis-dc-checkbox"]').check();
  await expect(page.locator('[data-testid="trellis-delta-dc"]')).toBeVisible();
  await page.locator('[data-testid="trellis-lambda1"]').fill("0");
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getCurrent()?.samplingFactors)).toEqual([[2, 2], [1, 1], [1, 1]]);
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().trellisLambdaLogScale1)).toBe(0);
  await page.locator('label:has([data-testid="mode-hybrid"])').click();
  await expect(page.locator('#trellis-settings')).toBeHidden();
  await expect(page.locator('#hybrid-settings')).toBeVisible();
  await page.locator('[data-testid="aq-checkbox"]').uncheck();
  await page.locator('#hybrid-settings summary').click();
  await expect(page.locator('#hybrid-aq-note')).toBeVisible();
  await page.locator('[data-testid="hybrid-aq-exp"]').fill("0");
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().hybridAqExponent)).toBe(0);
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getLastEncode()?.mode)).toBe("hybrid");
  await page.locator('label:has([data-testid="mode-trellis"])').click();
  await expect(page.locator('[data-testid="trellis-lambda1"]')).toHaveValue("0");
});

test("color choices expose the corresponding sampling controls", async ({ page }) => {
  await page.locator('label:has(input[name="sub-choice"][value="halfHorizontal"])').click();
  await page.locator('label:has(input[name="color-choice"][value="xyb"])').click();
  await expect(page.locator('[data-testid="subsampling-options"]')).toBeHidden();
  await expect(page.locator('[data-testid="xyb-subsampling-options"]')).toBeVisible();
  await expect(page.locator('[data-testid="xyb-sampling-resolved"]')).toContainText("B quarter");
  await page.locator('.settings-details summary').filter({hasText:"More encoder settings"}).click();
  await expect(page.locator('[data-testid="sharp-yuv-checkbox"]')).toBeHidden();
  await page.locator('label:has(input[name="color-choice"][value="ycbcr"])').click();
  await expect(page.locator('input[name="sub-choice"][value="halfHorizontal"]')).toBeChecked();
  await page.locator('label:has(input[name="sub-choice"][value="none"])').click();
  await expect(page.locator('[data-testid="sharp-yuv-checkbox"]')).toBeHidden();
});

test("auto defaults and restart choices resolve to codec output", async ({ page }) => {
  await expect(page.locator('input[name="color-choice"][value="auto"]')).toBeChecked();
  await expect(page.locator('input[name="sub-choice"][value="auto"]')).toBeChecked();
  await expect(page.locator('input[name="scan"][value="auto"]')).toBeChecked();
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurrent()?.samplingFactors)).toEqual([[1, 1], [1, 1], [1, 1]]);
  await page.locator('label:has(input[name="scan"][value="restart"])').click();
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().restartMcuRows)).toBeUndefined();
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().progressive)).toBe(false);
  await page.evaluate(() => {document.body.dataset["encodeState"] = "waiting"});
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  const interval = await page.evaluate(() => window.__zenjpegDiagnostics.getCurrent()!.restartIntervalMcus);
  await expect(page.locator('[data-testid="scan-resolved"]')).toContainText(`restart interval ${interval} MCUs`);
  await page.locator('label:has(input[name="scan"][value="sequential"])').click();
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().restartMcuRows)).toBe(0);
});

test("a regenerated curve freezes custom settings and varies its quality-derived tables", async ({ page }) => {
  await page.locator('label:has(input[name="sub-choice"][value="quarter"])').click();
  await page.locator('[data-testid="quality-slider"]').evaluate(el => {
    const input = el as HTMLInputElement; input.value = "3"; input.dispatchEvent(new Event("input", {bubbles:true}));
  });
  await page.evaluate(() => {document.body.dataset["encodeState"] = "waiting"});
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  await page.locator('[data-testid="tab-quant"]').click();
  await page.locator('[data-testid="comp-0-h-eq-0"]').evaluate(el => {
    const input = el as HTMLInputElement; input.value = "2"; input.dispatchEvent(new Event("input", {bubbles:true}));
  });
  await page.locator('.quant-zero-bias summary').click();
  await page.locator('#zb-dc-0').fill("0.3");
  await page.evaluate(() => {document.body.dataset["encodeState"] = "waiting"});
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  const currentSize = await page.evaluate(() => window.__zenjpegDiagnostics.getLastEncode()!.bytes);
  await page.locator('[data-testid="controls-close"]').click();
  await page.locator('[data-testid="rd-regenerate"]').click();
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getCurve()?.points.some(p => p.q === 3)), {timeout:30000}).toBe(true);
  const curve = await page.evaluate(() => window.__zenjpegDiagnostics.getCurve()!);
  expect(curve.options.subsampling).toBe("quarter");
  expect(curve.custom).toBe(true);
  expect(curve.eq[0]!.hEq[0]).toBe(2);
  expect(curve.bias.offsetDc[0]).toBeCloseTo(0.3);
  expect(curve.points.find(p => p.q === 3)?.bytes).toBe(currentSize);
  await expect(page.locator('[data-testid="rd-canvas"]')).toHaveAttribute("aria-label", /bytes.*SSIMULACRA 2/);
  await page.locator('[data-testid="controls-toggle"]').click();
  await page.locator('[data-testid="tab-aq"]').click();
  await page.locator('label:has(input[name="sub-choice"][value="none"])').click();
  expect(await page.evaluate(() => window.__zenjpegDiagnostics.getCurve()!.options.subsampling)).toBe("quarter");
});

test("Butteraugli is selectable and matches the curve at the same settings", async ({ page }) => {
  await page.locator('[data-testid="quality-slider"]').evaluate(el => {
    const input = el as HTMLInputElement; input.value = "3"; input.dispatchEvent(new Event("input", {bubbles:true}));
  });
  await page.evaluate(() => {document.body.dataset["encodeState"] = "waiting"});
  await page.locator('[data-testid="encode-button"]').click();
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  await page.locator('[data-testid="controls-close"]').click();
  await page.locator('[data-testid="rd-metric"]').selectOption("butteraugli");
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getCurve()?.points.some(p => p.q === 3)), {timeout:30000}).toBe(true);
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getLastEncode()?.butteraugli)).not.toBeNull();
  const {curve, current} = await page.evaluate(() => ({curve: window.__zenjpegDiagnostics.getCurve()!, current: window.__zenjpegDiagnostics.getLastEncode()!}));
  expect(curve.metric).toBe("butteraugli");
  expect(curve.points[0]!.score).toBeGreaterThanOrEqual(0);
  expect(curve.points.find(p => p.q === 3)?.bytes).toBe(current.bytes);
  expect(curve.points.find(p => p.q === 3)!.score).toBeCloseTo(current.butteraugli!, 5);
  await expect(page.locator('[data-testid="rd-canvas"]')).toHaveAttribute("aria-label", /Butteraugli distance/);
  await expect(page.locator('[data-testid="rd-legend"]')).toContainText("White");
  await expect(page.locator('#m-butter')).not.toHaveText("—");
});

test("an encode preserves the focused slider and an open cell editor", async ({ page }) => {
  await page.locator('[data-testid="tab-quant"]').click();
  const slider = page.locator('[data-testid="comp-0-h-eq-0"]');
  await slider.focus();
  await slider.evaluate((el) => {
    const input = el as HTMLInputElement;
    input.value = "1.5";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    document.body.dataset["encodeState"] = "waiting";
    document.querySelector<HTMLButtonElement>('[data-testid="encode-button"]')!.click();
  });
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  await expect(slider).toBeFocused();
  await page.locator('[data-testid="comp-0-cell-0"]').click();
  const input = page.locator('[data-testid="cell-editor-value"]');
  await input.fill("37");
  await page.evaluate(() => {
    document.body.dataset["encodeState"] = "waiting";
    document.querySelector<HTMLButtonElement>('[data-testid="encode-button"]')!.click();
  });
  await expect.poll(() => page.evaluate(() => document.body.dataset["encodeState"])).toBe("done");
  await expect(input).toBeVisible();
  await expect(input).toHaveValue("37");
  await page.locator('[data-testid="cell-editor-apply"]').click();
  await expect.poll(() => page.evaluate(() => window.__zenjpegDiagnostics.getCurrentOptions().customQuantTables?.y[0])).toBe(37);
});

test("keyboard zoom restores the settings panel", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const canvas = page.locator('[data-testid="source-canvas"]');
  await canvas.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator('[data-testid="zoom-modal"]')).toBeVisible();
  await expect(page.locator('[data-testid="controls-popover"]')).toBeHidden();
  await page.keyboard.press("Escape");
  await expect(page.locator('[data-testid="zoom-modal"]')).toBeHidden();
  await expect(page.locator('[data-testid="controls-popover"]')).toBeVisible();
  await expect(canvas).toBeFocused();
});

for (const [width, height] of [[320, 700], [390, 844], [768, 1024], [1440, 900], [844, 390]]) {
  test(`responsive controls remain usable at ${width}×${height}`, async ({ page }) => {
    await page.setViewportSize({ width: width!, height: height! });
    await page.locator('[data-testid="tab-quant"]').click();
    await page.locator('[data-testid="component-pick-2"]').click();
    const cell = page.locator('[data-testid="comp-2-cell-0"]');
    await cell.scrollIntoViewIfNeeded();
    await cell.click();
    const editor = page.locator('[data-testid="cell-editor"]');
    await expect(editor).toBeVisible();
    const bounds = await page.evaluate(() => {
      const panel = document.querySelector<HTMLElement>('#controls-popover')!.getBoundingClientRect();
      const popup = document.querySelector<HTMLElement>('#quant-cell-editor')!.getBoundingClientRect();
      const content = document.querySelector<HTMLElement>("#tab-panels")!;
      return { viewport: innerWidth, scroll: document.documentElement.scrollWidth, contentWidth: content.clientWidth, contentScroll: content.scrollWidth, panel: { left: panel.left, right: panel.right, top: panel.top, bottom: panel.bottom }, popup: { left: popup.left, right: popup.right, top: popup.top, bottom: popup.bottom }, height: innerHeight };
    });
    expect(bounds.scroll).toBeLessThanOrEqual(bounds.viewport);
    expect(bounds.contentScroll).toBeLessThanOrEqual(bounds.contentWidth);
    for (const rect of [bounds.panel, bounds.popup]) {
      expect(rect.left).toBeGreaterThanOrEqual(0);
      expect(rect.right).toBeLessThanOrEqual(bounds.viewport);
      expect(rect.top).toBeGreaterThanOrEqual(0);
      expect(rect.bottom).toBeLessThanOrEqual(bounds.height);
    }
    await page.keyboard.press("Escape");
    await expect(editor).toBeHidden();
    await expect(page.locator('[data-testid="controls-popover"]')).toBeVisible();
  });
}
