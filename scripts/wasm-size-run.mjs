// Run the size probe with JPEG stdin and dimensions + raster stdout.
// WASI API: https://nodejs.org/api/wasi.html
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
const wasi = new WASI({ version: 'preview1', args: ['wasm-size-probe'] });
const module = await WebAssembly.compile(await readFile(process.argv[2]));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);
