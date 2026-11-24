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

// mirrors the harness head(): layers take [c, s, g, mask]; graph models take named inputs. Outputs are told apart by width.
export function headRaw(f, maskArr) {
  const m = tf.tensor2d([maskArr]); let o;
  if (E.man.head.format === 'layers') o = E.H.predict([...f, m]);
  else { const by = { center: f[0], surround: f[1], global: f[2], mask: m }; o = E.H.execute(Object.fromEntries(E.man.head.inputs.map(n => [n, by[n]]))); }
  const a = Array.isArray(o) ? o : [o];
  return { lg: a.find(t => t.shape[1] === E.D.classes.length), g: a.find(t => t.shape[1] === 3), m };
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
  if (!man) throw new UserError('model/manifest.json was not found. Run “python export.py b4vd 42” and copy the web/ folder into model/.');
  if (!D?.classes) throw new UserError('model/data.json is missing or has no class list.');
  tf = await loadTF(); E.backend = tf.getBackend();
  const p = [0, 0, man.backbone_cam ? 0 : 1], rep = () => onProgress(0.08 + 0.87 * ((p[0] + p[1] + p[2]) / 3), 'Loading model weights');
  const [B, H, camM] = await Promise.all([
    ld(man.backbone, f => { p[0] = f; rep(); }), ld(man.head, f => { p[1] = f; rep(); }),
    man.backbone_cam ? ld(man.backbone_cam, f => { p[2] = f; rep(); }).catch(() => { p[2] = 1; rep(); return null; }) : null]);
  Object.assign(E, { man, D, B, H });
  if (camM) { E.CAM = { model: camM, format: man.backbone_cam.format, inputs: man.backbone_cam.inputs }; E.camHow = 'manifest'; }
  else { const sp = splitFromLayers(B, man); if (sp) { E.CAM = sp; E.camHow = 'derived'; } }
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