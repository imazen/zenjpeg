// Build-time image fetch + wash hook.
//
// Downloads the diagnostics-viewer demo corpus from a Cloudflare R2 (or
// any HTTP) endpoint into `public/images/`, then writes a
// `manifest.json` the runtime reads at boot.
//
// Two URL conventions per entry:
//   1. Content-addressed: entry has `sha256`. Path is
//      `<base>/blobs/<sha[0:2]>/<sha[2:4]>/<sha>`.
//   2. Path-addressed: entry has `file`. Path is `<base>/<file>`.
//
// Wash step (CRITICAL for diagnostics integrity):
//   When `wasJpeg: true`, the source image is decoded and 8×
//   downsampled with sharp's lanczos3 kernel before being re-encoded
//   as PNG. This eliminates any 8×8 DCT block artifacts inherited
//   from a JPEG → PNG conversion in the source corpus, so the encoded
//   side of the diagnostics is measuring our encoder, not someone
//   else's. Lanczos3 is the right kernel here — Mitchell-Netravali
//   is gentler but leaves more residual ringing; lanczos3 is sharper
//   with controlled overshoot. See:
//     ~/work/claudehints/topics/rust-defaults.md  ("principled kernel")
//
// The post-wash file is named `<sha>-w8.png` so the original blob can
// also be cached separately if desired (the runtime never needs the
// raw blob — it always reads the washed PNG via the manifest).
//
// Configuration env:
//   DIAGNOSTICS_VIEWER_IMAGE_BASE
//     base URL prefix (no trailing slash). Defaults to the source
//     manifest's `defaultBase`, then a baked-in codec-corpus URL.
//   DIAGNOSTICS_VIEWER_IMAGE_LIST
//     path to the source manifest. Defaults to ./scripts/image-list.json.
//   DIAGNOSTICS_VIEWER_OFFLINE
//     "1" to skip network entirely; reuse existing local files.
//
// Run as `npm run prebuild` or `tsx scripts/fetch-images.ts`. The
// only runtime deps are sharp (for the wash step) and Node's
// built-in fetch.

import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import sharp from "sharp";

interface SourceEntry {
  id: string;
  label: string;
  /** Optional content-addressed sha256 hex. */
  sha256?: string;
  /** File extension to use for the raw cached copy. */
  ext?: string;
  /** Plain relative URL fragment (alternative to `sha256`). */
  file?: string;
  /** When true, decode + 8× lanczos3 downscale + re-encode as PNG.
   *  Use for sources that may have come from JPEG (the codec-corpus
   *  doesn't track provenance, so its `corpus/png-24-32` PNGs are
   *  conservatively flagged). */
  wasJpeg?: boolean;
}

interface SourceManifest {
  defaultBase?: string;
  images: SourceEntry[];
}

interface RuntimeEntry {
  id: string;
  label: string;
  url: string;
  /** Post-wash dimensions, included in the manifest so the runtime
   *  can size its UI without decoding. */
  width: number;
  height: number;
  /** True when the served file is the post-wash PNG (vs the raw
   *  blob). The runtime uses this for status-line annotation. */
  washed: boolean;
}

const FALLBACK_DEFAULT_BASE =
  "https://pub-7c5c57fd3e0842f0b147946928891d40.r2.dev";

const here = dirname(new URL(import.meta.url).pathname);
const webRoot = resolve(here, "..");
const targetDir = resolve(webRoot, "public", "images");
const manifestSrc = resolve(
  process.env.DIAGNOSTICS_VIEWER_IMAGE_LIST ??
    resolve(here, "image-list.json"),
);
const manifestOut = resolve(targetDir, "manifest.json");
const offline = process.env.DIAGNOSTICS_VIEWER_OFFLINE === "1";

function readSource(): SourceManifest {
  if (!existsSync(manifestSrc)) {
    console.warn(
      `[fetch-images] no source manifest at ${manifestSrc}; nothing to fetch`,
    );
    return { images: [] };
  }
  const raw = readFileSync(manifestSrc, "utf8");
  try {
    const data = JSON.parse(raw) as SourceManifest;
    if (!Array.isArray(data.images)) {
      console.warn("[fetch-images] manifest source missing `images` array");
      return { images: [] };
    }
    return data;
  } catch (e) {
    console.warn(`[fetch-images] failed to parse ${manifestSrc}:`, e);
    return { images: [] };
  }
}

function sha256Hex(buf: Uint8Array): string {
  const h = createHash("sha256");
  h.update(buf);
  return h.digest("hex");
}

interface BlobAddr {
  /** URL path component, joined onto base with a slash. */
  pathFragment: string;
  /** Raw-blob filename. */
  rawName: string;
  /** Final served filename (raw or post-wash, depending on wasJpeg). */
  servedName: string;
}

function deriveAddr(entry: SourceEntry): BlobAddr | null {
  if (entry.sha256) {
    const sha = entry.sha256.toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(sha)) {
      console.warn(`[fetch-images] invalid sha256 for ${entry.id}: ${sha}`);
      return null;
    }
    const ext = entry.ext ?? "bin";
    const rawName = `${sha}.${ext}`;
    const servedName = entry.wasJpeg ? `${sha}-w8.png` : rawName;
    return {
      pathFragment: `blobs/${sha.slice(0, 2)}/${sha.slice(2, 4)}/${sha}`,
      rawName,
      servedName,
    };
  }
  if (entry.file) {
    const sansExt = entry.file.replace(/\.[^./]+$/, "");
    const rawName = entry.file;
    const servedName = entry.wasJpeg ? `${sansExt}-w8.png` : entry.file;
    return {
      pathFragment: entry.file,
      rawName,
      servedName,
    };
  }
  console.warn(
    `[fetch-images] entry ${entry.id} has neither sha256 nor file; skipping`,
  );
  return null;
}

async function washToPng(
  rawPath: string,
  outPath: string,
): Promise<{ width: number; height: number }> {
  // 8× downscale via sharp lanczos3, written as PNG. sharp accepts
  // the source format directly (PNG/JPEG/WebP/AVIF/etc.) and the
  // resize() default kernel is lanczos3. The PNG output is lossless
  // (compressionLevel default 6 is fine — these are <2MB demo blobs).
  const meta = await sharp(rawPath).metadata();
  const w0 = meta.width ?? 0;
  const h0 = meta.height ?? 0;
  if (w0 === 0 || h0 === 0) {
    throw new Error(`sharp could not read dimensions of ${rawPath}`);
  }
  const w = Math.max(8, Math.floor(w0 / 8));
  const h = Math.max(8, Math.floor(h0 / 8));
  await sharp(rawPath)
    .resize(w, h, { kernel: "lanczos3" })
    .png({ compressionLevel: 9, palette: false })
    .toFile(outPath);
  return { width: w, height: h };
}

async function dimsOnly(path: string): Promise<{ width: number; height: number }> {
  const meta = await sharp(path).metadata();
  return { width: meta.width ?? 0, height: meta.height ?? 0 };
}

async function fetchOne(
  entry: SourceEntry,
  baseUrl: string,
): Promise<RuntimeEntry | null> {
  const addr = deriveAddr(entry);
  if (!addr) return null;
  const rawDest = resolve(targetDir, addr.rawName);
  const servedDest = resolve(targetDir, addr.servedName);
  const wasJpeg = !!entry.wasJpeg;

  // Decide whether we need to download. The served file (washed or
  // raw) is what the runtime consumes. For wash entries we keep the
  // raw blob too so we can re-wash with a new kernel without
  // re-downloading.
  if (offline) {
    if (!existsSync(servedDest)) {
      console.warn(
        `[fetch-images] offline + missing served file: ${addr.servedName}`,
      );
      return null;
    }
    const dims = await dimsOnly(servedDest);
    return {
      id: entry.id,
      label: entry.label,
      url: `./images/${addr.servedName}`,
      width: dims.width,
      height: dims.height,
      washed: wasJpeg,
    };
  }
  if (!baseUrl) {
    if (existsSync(servedDest)) {
      const dims = await dimsOnly(servedDest);
      return {
        id: entry.id,
        label: entry.label,
        url: `./images/${addr.servedName}`,
        width: dims.width,
        height: dims.height,
        washed: wasJpeg,
      };
    }
    console.warn(
      `[fetch-images] no base URL and no local copy for ${entry.id}; skipping`,
    );
    return null;
  }

  // Reuse cached raw blob when sha matches (or no hash declared and
  // we already have it on disk).
  let needsDownload = true;
  if (existsSync(rawDest)) {
    if (!entry.sha256) {
      needsDownload = false;
    } else {
      const have = readFileSync(rawDest);
      if (sha256Hex(have) === entry.sha256.toLowerCase()) {
        needsDownload = false;
      } else {
        console.log(
          `[fetch-images] sha256 mismatch on cached ${addr.rawName}, re-downloading`,
        );
      }
    }
  }
  if (needsDownload) {
    const url = `${baseUrl}/${addr.pathFragment}`;
    console.log(`[fetch-images] GET ${url}`);
    const resp = await fetch(url);
    if (!resp.ok) {
      console.warn(`[fetch-images] ${url} → ${resp.status} ${resp.statusText}`);
      return null;
    }
    const buf = new Uint8Array(await resp.arrayBuffer());
    if (entry.sha256 && sha256Hex(buf) !== entry.sha256.toLowerCase()) {
      console.warn(`[fetch-images] sha256 mismatch on download from ${url}`);
      return null;
    }
    mkdirSync(dirname(rawDest), { recursive: true });
    writeFileSync(rawDest, buf);
  }

  // Wash if needed (and the wash output isn't already on disk and fresh).
  let dims: { width: number; height: number };
  if (wasJpeg) {
    const rawStat = needsDownload || !existsSync(servedDest);
    if (rawStat) {
      console.log(
        `[fetch-images] washing ${addr.rawName} → ${addr.servedName} (8× lanczos3 → PNG)`,
      );
      dims = await washToPng(rawDest, servedDest);
    } else {
      dims = await dimsOnly(servedDest);
    }
  } else {
    dims = await dimsOnly(rawDest);
  }
  return {
    id: entry.id,
    label: entry.label,
    url: `./images/${addr.servedName}`,
    width: dims.width,
    height: dims.height,
    washed: wasJpeg,
  };
}

async function main(): Promise<void> {
  mkdirSync(targetDir, { recursive: true });
  const source = readSource();
  if (source.images.length === 0) {
    writeFileSync(manifestOut, JSON.stringify({ images: [] }, null, 2) + "\n");
    console.log(`[fetch-images] wrote empty manifest at ${manifestOut}`);
    return;
  }
  const baseUrl = (
    process.env.DIAGNOSTICS_VIEWER_IMAGE_BASE ??
    source.defaultBase ??
    FALLBACK_DEFAULT_BASE
  ).replace(/\/+$/, "");
  console.log(`[fetch-images] base URL: ${baseUrl || "(none — offline)"}`);
  const runtime: RuntimeEntry[] = [];
  for (const entry of source.images) {
    try {
      const got = await fetchOne(entry, baseUrl);
      if (got) runtime.push(got);
    } catch (e) {
      console.warn(`[fetch-images] error on ${entry.id}:`, e);
    }
  }
  writeFileSync(
    manifestOut,
    JSON.stringify({ images: runtime }, null, 2) + "\n",
  );
  console.log(
    `[fetch-images] wrote runtime manifest with ${runtime.length}/${source.images.length} entries at ${manifestOut}`,
  );
}

main().catch((e) => {
  console.error("[fetch-images] fatal:", e);
  process.exitCode = 1;
});
