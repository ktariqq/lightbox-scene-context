import { CONFIG as C } from './config.js';
import { UserError } from './util.js';
import { exportURL } from './data.js';

const canvas = w => Object.assign(document.createElement('canvas'), { width: w, height: w });
const ctx2 = c => { const x = c.getContext('2d', { willReadFrequently: true }); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high'; return x; };

// halve until within 2x of the target, then one final draw: avoids aliasing on large photos
function resample(src, sx, sy, side, size) {
  let cur = src, x = sx, y = sy, w = side;
  while (w > size * 2) { const nw = Math.ceil(w / 2), c = canvas(nw); ctx2(c).drawImage(cur, x, y, w, w, 0, 0, nw, nw); cur = c; x = y = 0; w = nw; }
  const out = canvas(size); ctx2(out).drawImage(cur, x, y, w, w, 0, 0, size, size); return out;
}

function lumStd(c) {
  const t = canvas(64), x = ctx2(t); x.drawImage(c, 0, 0, 64, 64);
  const d = x.getImageData(0, 0, 64, 64).data; let a = 0, q = 0;
  for (let i = 0; i < d.length; i += 4) { const l = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]; a += l; q += l * l; }
  const m = a / 4096; return Math.sqrt(Math.max(0, q / 4096 - m * m));
}

export async function decode(file) {
  if (!C.ACCEPT.includes(file.type)) throw new UserError(`“${file.name}” is ${file.type || 'an unknown file type'}. Please use a JPEG, PNG or WebP image.`);
  if (file.size > C.MAX_FILE_MB * 1048576) throw new UserError(`That file is ${(file.size / 1048576).toFixed(1)} MB. The limit is ${C.MAX_FILE_MB} MB. Export a smaller JPEG and try again.`);
  let src;
  try { src = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
  catch {
    try { const img = new Image(); img.src = URL.createObjectURL(file); await img.decode(); URL.revokeObjectURL(img.src); src = img; }
    catch { throw new UserError(`“${file.name}” could not be read as an image. The file may be damaged or in an unsupported variant.`); }
  }
  const w = src.width ?? src.naturalWidth, h = src.height ?? src.naturalHeight;
  if (!w || !h) throw new UserError('The image has no readable pixels.');
  if (w * h > 120e6) { src.close?.(); throw new UserError(`That image is ${w} × ${h} px, which is too large for the browser to handle safely. Please downsize it first.`); }
  return { src, w, h };
}

// centre-crop to a square, resample to 1280, cut the nested boxes from CONFIG.VIEWS, resample each to 224
export function buildViews(src, w, h) {
  const side = Math.min(w, h), ox = Math.floor((w - side) / 2), oy = Math.floor((h - side) / 2);
  const full = resample(src, ox, oy, side, C.SIDE), views = {};
  for (const v of C.VIEWS) views[v.id] = resample(full, v.box[0], v.box[1], v.box[2] - v.box[0], C.INPUT);
  return { views, info: { w, h, aspect: Math.max(w, h) / Math.min(w, h), minSide: side, std: lumStd(views.global) } };
}

const loadImg = src => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new UserError(`Could not load ${src}`)); i.src = src; });
export async function viewsFromItem(item) {
  const out = {};
  await Promise.all(C.VIEWS.map(async v => { const img = await loadImg(exportURL(item.views[v.id].image)); const c = canvas(C.INPUT); ctx2(c).drawImage(img, 0, 0, C.INPUT, C.INPUT); out[v.id] = c; }));
  return out;
}