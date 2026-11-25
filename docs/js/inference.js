import { CONFIG as C } from './config.js';
import { UserError } from './util.js';
import { exportURL, loadManifest, loadData, loadParity } from './data.js';

let tf = null, loading = null;
export const getTF = () => tf;
export const E = { ready: false, backend: '—', man: null, D: null, B: null, H: null, CAM: null, camHow: 'none' };

const script = src => new Promise((res, rej) => { const el = document.createElement('script'); el.src = src; el.onload = res; el.onerror = () => rej(new Error('Could not load ' + src)); document.head.append(el); });

async function loadTF() {
  if (!window.tf) await script(C.TF.core);
  const t = window.tf;
  const ok = async n => { try { return (await t.setBackend(n)) && (await t.ready(), t.getBackend() === n); } catch { return false; } };
  if (!(await ok('webgl'))) {
    try { await script(C.TF.wasm); const w = window.tf.wasm ?? window.tf; (w.setWasmPaths ?? w.setWasmPath)?.call(w, C.TF.wasmPaths); } catch { /* fall through to cpu */ }
    if (!(await ok('wasm'))) await ok('cpu');
  }
  return t;
}

const ld = (spec, onProgress) => { const url = exportURL(spec.path); return spec.format === 'layers' ? tf.loadLayersModel(url, { onProgress }) : tf.loadGraphModel(url, { onProgress }); };

// derive the split backbone (conv map + pooled) from a Keras-layers backbone when the export has no dedicated one
function splitFromLayers(B, man) {
  if (man.backbone.format !== 'layers') return null;
  try { const l = B.getLayer('out_relu'); return { model: tf.model({ inputs: B.inputs, outputs: [l.output, B.outputs[0]] }), format: 'layers' }; } catch { return null; }
}

// harness feeds raw 0..255 floats, so the exported backbone must contain its own preprocessing
function backbone(x) {
  const u = E.CAM ?? { model: E.B, format: E.man.backbone.format, inputs: E.man.backbone.inputs };
  const r = u.format === 'layers' ? u.model.predict(x) : u.model.execute({ [u.inputs[0]]: x });
  const a = Array.isArray(r) ? r : [r], pooled = a.find(t => t.rank === 2), conv = a.find(t => t.rank === 4);
  if (!pooled) throw new Error('the backbone returned no pooled feature vector');
  return { pooled, conv };
}

// Load the head weights dumped by export_web (head.json + head.bin) and return a run function.
// Mirrors the MOSAIC example: a ~12-line tfjs forward pass over the raw float weights.
export async function loadHead(base = './') {
  const m = await (await fetch(exportURL('head.json'))).json();
  const buf = new Float32Array(await (await fetch(exportURL('head.bin'))).arrayBuffer());
  const T = {};
  for (const [k, v] of Object.entries(m.w))
    T[k] = tf.tensor(buf.slice(v.offset, v.offset + v.shape.reduce((a, c) => a * c, 1)), v.shape);
  const d = (x, n, a) => { const y = x.matMul(T[n + '/k']).add(T[n + '/b']); return a ? a(y) : y; };
  return (feat, mask) => tf.tidy(() => {
    const n = mask.reduce((a, c) => a + c);
    const pm = m.views.map((v, i) => d(feat[v], 'proj_' + v, tf.relu).mul(mask[i]));
    const fl = tf.concat(pm, 1);
    const g = m.views.map((v, i) => d(fl, 'gate_' + v, tf.sigmoid).mul(mask[i]));
    return { logits: d(d(tf.addN(g.map((x, i) => x.mul(pm[i]))).div(n), 'hid', tf.relu), 'logits'),
             gates: g.map(x => x.mean().dataSync()[0]) };
  });
}

// mirrors the harness head(): now backed by the raw-weight head from export_web.
export function headRaw(f, maskArr) {
  const names = ['center', 'surround', 'global'];
  const feat = Array.isArray(f)
    ? Object.fromEntries(names.map((v, i) => [v, f[i]]))
    : f;
  const { logits, gates } = E.H(feat, maskArr);
  return { lg: logits, g: tf.tensor1d(gates), m: null };
}

const softmax = a => { const mx = Math.max(...a), e = Array.from(a, x => Math.exp(x - mx)), z = e.reduce((p, q) => p + q, 0); return e.map(x => x / z); };

export async function runHead(feat, mask) {
  const o = tf.tidy(() => { const { lg, g } = headRaw(feat.pooled, mask); return { lg, g }; });
  const [lg, g] = await Promise.all([o.lg.data(), o.g ? o.g.data() : null]); tf.dispose(o);
  return { logits: lg, probs: softmax(lg), gates: g };
}

export async function extract(views) {
  const t0 = performance.now(), pooled = [], conv = [];
  for (const v of C.VIEWS) {
    const o = tf.tidy(() => backbone(tf.browser.fromPixels(views[v.id]).toFloat().expandDims(0)));
    pooled.push(o.pooled); conv.push(o.conv);
    await tf.nextFrame(); // keep the page responsive between views
  }
  await Promise.all(pooled.map(t => t.data())); // force the GPU work to finish so the timing is honest
  return { pooled, conv: conv.every(Boolean) ? conv : null, ms: performance.now() - t0 };
}
export const disposeFeat = f => f && tf.dispose([...f.pooled, ...(f.conv ?? [])]);

async function warm() {
  const z = tf.zeros([1, C.INPUT, C.INPUT, 3]), o = backbone(z); z.dispose();
  await runHead({ pooled: [o.pooled, o.pooled, o.pooled] }, [1, 1, 1]);
  tf.dispose([o.pooled, o.conv].filter(Boolean));
}

async function run(onProgress) {
  onProgress(0.02, 'Loading TensorFlow.js');
  const [man, D] = await Promise.all([loadManifest(), loadData()]);
  if (!man) throw new UserError('model/manifest.json was not found. Run “python export.py b4vd 42”.');
  if (!D?.classes) throw new UserError('model/data.json is missing or has no class list.');
  tf = await loadTF(); E.backend = tf.getBackend();
  const p = [0, 1], rep = () => onProgress(0.08 + 0.87 * ((p[0] + p[1]) / 2), 'Loading model weights');
  const B = await ld(man.backbone, f => { p[0] = f; rep(); });
  const head = await loadHead('./');   // head.bin + head.json from export_web (numpy-verified forward pass)
  p[1] = 1; rep();
  Object.assign(E, { man, D, B, H: head });
  // No TF.js cam model exported; Grad-CAM overlays from export.py (cam/) are used in evidence instead.
  const sp = splitFromLayers(B, man); if (sp) { E.CAM = sp; E.camHow = 'derived'; }
  onProgress(0.96, 'Warming up'); await warm(); E.ready = true; onProgress(1, 'Ready');
}

export function ensureModel(cb = () => {}) {
  if (!loading) loading = run(cb).catch(e => { loading = null; e.shown = false; throw e; });
  return loading;
}

export async function runParity() {
  const p = await loadParity();
  if (!p) throw new UserError('parity.json is missing from model/. Run “python export.py b4vd 42” and copy web/ into model/.');
  const f = C.VIEWS.map(v => tf.tensor2d([p[v.id]])), lines = []; let worst = 0;
  for (const c of p.cases) {
    const o = tf.tidy(() => { const { lg, g } = headRaw(f, c.mask); return { lg, g }; });
    const [lg, g] = await Promise.all([o.lg.data(), o.g ? o.g.data() : null]); tf.dispose(o);
    const dl = Math.max(...c.logits.map((x, i) => Math.abs(x - lg[i]))), dg = c.gates && g ? Math.max(...c.gates.map((x, i) => Math.abs(x - g[i]))) : NaN;
    lines.push(`mask ${c.mask.join('')}: max|dlogits|=${dl.toExponential(2)}  max|dgates|=${dg.toExponential(2)}`);
    worst = Math.max(worst, dl, Number.isNaN(dg) ? 0 : dg);
  }
  tf.dispose(f);
  lines.push('', `backend ${E.backend} · worst difference ${worst.toExponential(2)}`, 'Same two numbers per mask as the export harness prints. Judge them against your Keras tolerance.');
  return lines;
}