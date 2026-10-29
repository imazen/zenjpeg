// Download the optional demo sources unchanged. No generated images are
// checked into the repository. Existing files are verified, never overwritten.
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const manifest = JSON.parse(await readFile(resolve(import.meta.dirname, "image-list.json"), "utf8")) as {
  defaultBase: string;
  images: { id: string; sha256: string; ext: string }[];
};
const base = (process.env.DIAGNOSTICS_VIEWER_IMAGE_BASE ?? manifest.defaultBase).replace(/\/+$/, "");
const dir = resolve(root, "public/images");
await mkdir(dir, { recursive: true });
for (const entry of manifest.images) {
  const sha = entry.sha256.toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(sha) || !/^[a-z0-9]+$/.test(entry.ext)) throw new Error(`Invalid source: ${entry.id}`);
  const path = resolve(dir, `${sha}.${entry.ext}`);
  const url = `${base}/blobs/${sha.slice(0, 2)}/${sha.slice(2, 4)}/${sha}`;
  let bytes: Uint8Array;
  try {
    bytes = await readFile(path);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    bytes = new Uint8Array(await response.arrayBuffer());
    if (createHash("sha256").update(bytes).digest("hex") !== sha) throw new Error(`Hash mismatch: ${url}`);
    await writeFile(path, bytes, { flag: "wx" });
  }
  if (createHash("sha256").update(bytes).digest("hex") !== sha) throw new Error(`Existing file has wrong hash: ${path}`);
  console.log(`Ready: ${entry.id}`);
}
