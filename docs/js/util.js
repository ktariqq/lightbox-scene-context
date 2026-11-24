export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const REDUCED = matchMedia('(prefers-reduced-motion: reduce)');
export class UserError extends Error {}

export const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'class') e.className = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
};
export const s = (tag, attrs = {}, ...kids) => {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null && v !== false) e.setAttribute(k, v);
  for (const c of kids.flat(Infinity)) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
};

export const resolve = (p, root = '') => (/^(https?:|data:|blob:|\/)/.test(p) ? p : new URL(root + p, document.baseURI).href);
export const fmt = (v, d = 3) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
export const pm = (m, sd, d = 3) => (typeof m === 'number' ? `${fmt(m, d)} ± ${fmt(sd ?? 0, d)}` : '—');
export const norm = n => String(n).toLowerCase().replace(/[^a-z0-9]/g, '');
const LABELS = { majority: 'Majority', chance: 'Chance', customcnn: 'B1 custom CNN', b2: 'B2 frozen MobileNetV2 (center)', b3: 'B3 concat', b3vd: 'B3-VD', b4: 'B4 gated', b4vd: 'B4-VD (proposed)' };
export const modelLabel = n => LABELS[norm(n)] ?? n;
export const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
export const sd = a => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
export const groupBy = (rows, f) => { const m = new Map(); for (const r of rows) { const k = f(r); (m.get(k) ?? m.set(k, []).get(k)).push(r); } return m; };
export const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };
export const nextFrame = () => new Promise(r => requestAnimationFrame(() => r()));

const bus = new EventTarget();
export const emit = (n, d) => bus.dispatchEvent(new CustomEvent(n, { detail: d }));
export const on = (n, f) => bus.addEventListener(n, e => f(e.detail));