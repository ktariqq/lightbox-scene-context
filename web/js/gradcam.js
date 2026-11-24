import { CONFIG as C } from './config.js';
import { getTF, headRaw } from './inference.js';

const RAMPS = {
  gold: [[0, [102, 51, 153]], [.35, [167, 123, 208]], [.65, [242, 193, 78]], [1, [255, 255, 255]]],
  cb: [[0, [68, 1, 84]], [.25, [59, 82, 139]], [.5, [33, 144, 141]], [.75, [93, 201, 99]], [1, [253, 231, 37]]], // viridis stops: colour-blind safe
};
const lut = name => {
  const st = RAMPS[name], out = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 1; while (k < st.length - 1 && t > st[k][0]) k++;
    const [a, ca] = st[k - 1], [b, cb] = st[k], f = Math.min(1, Math.max(0, (t - a) / (b - a)));
    for (let j = 0; j < 3; j++) out[i * 4 + j] = ca[j] + (cb[j] - ca[j]) * f;
    out[i * 4 + 3] = Math.min(1, Math.max(0, (t - 0.08) / 0.5)) * 235; // low values stay see-through, high values bloom
  }
  return out;
};
const LUTS = { gold: lut('gold'), cb: lut('cb') };

// Grad-CAM: d(logit of class)/d(conv map) through global average pooling and the head, channel-mean weights, ReLU, per-view max normalisation
export async function liveCam(feat, mask, cls) {
  const tf = getTF(); if (!feat.conv) return null;
  const maps = tf.tidy(() => {
    const f = (a, b, c) => { const { lg } = headRaw([a, b, c].map(t => t.mean([1, 2])), mask); return lg.slice([0, cls], [1, 1]).reshape([]); };
    const gr = tf.grads(f)(feat.conv);
    return feat.conv.map((A, i) => A.mul(gr[i].mean([1, 2], true)).sum(3).relu().squeeze());
  });
  const data = await Promise.all(maps.map(t => t.data())), size = maps[0].shape[0]; tf.dispose(maps);
  return data.map((d, i) => {
    if (!mask[i]) return null;
    const mx = Math.max(...d), out = new Float32Array(d.length); if (mx > 0) for (let k = 0; k < d.length; k++) out[k] = d[k] / mx;
    return { size, data: out };
  });
}

export function camCanvas(cam, ramp = 'gold') {
  const n = cam.size, small = Object.assign(document.createElement('canvas'), { width: n, height: n }), sx = small.getContext('2d');
  const id = sx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) { id.data[i * 4] = cam.data[i] * 255; id.data[i * 4 + 3] = 255; }
  sx.putImageData(id, 0, 0);
  const big = Object.assign(document.createElement('canvas'), { width: C.INPUT, height: C.INPUT }), bx = big.getContext('2d', { willReadFrequently: true });
  bx.imageSmoothingEnabled = true; bx.imageSmoothingQuality = 'high'; bx.drawImage(small, 0, 0, C.INPUT, C.INPUT);
  const px = bx.getImageData(0, 0, C.INPUT, C.INPUT), L = LUTS[ramp];
  for (let i = 0; i < px.data.length; i += 4) { const v = px.data[i] * 4; px.data[i] = L[v]; px.data[i + 1] = L[v + 1]; px.data[i + 2] = L[v + 2]; px.data[i + 3] = L[v + 3]; }
  bx.putImageData(px, 0, 0); return big;
}