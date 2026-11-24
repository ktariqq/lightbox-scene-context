import { h, s, fmt } from './util.js';

export const empty = f => h('p', { class: 'empty' }, `results not exported yet · ${f}`);
export const sel = (label, opts, val, on) => h('label', {}, label + ' ',
  h('select', { onchange: e => on(e.target.value) }, opts.map(o => h('option', { value: o.v, selected: String(o.v) === String(val) }, o.t))));
export const chk = (label, val, on) => h('label', {}, h('input', { type: 'checkbox', checked: !!val, onchange: e => on(e.target.checked) }), ' ' + label);

const THEMES = {
  dark: { cols: ['#A9E8E8', '#F2C14E', '#E8A7D1', '#CDB6EA', '#ffffff'], line: '#0A0614' },
  paper: { cols: ['#663399', '#F2C14E', '#E8A7D1', '#2A1540', '#A9E8E8'], line: '#2A1540' },
};
const SHAPES = ['circle', 'square', 'diamond', 'triangle', 'circle'];

function mark(shape, x, y, r, fill, line) {
  const a = { fill, stroke: line, 'stroke-width': 1.3 };
  if (shape === 'square') return s('rect', { ...a, x: x - r, y: y - r, width: 2 * r, height: 2 * r });
  if (shape === 'diamond') return s('polygon', { ...a, points: `${x},${y - r * 1.3} ${x + r * 1.3},${y} ${x},${y + r * 1.3} ${x - r * 1.3},${y}` });
  if (shape === 'triangle') return s('polygon', { ...a, points: `${x},${y - r * 1.3} ${x + r * 1.2},${y + r} ${x - r * 1.2},${y + r}` });
  return s('circle', { ...a, cx: x, cy: y, r });
}
function ticks(lo, hi, n = 5) {
  const span = hi - lo || Math.abs(hi) || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, out = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 0.5; v += step) out.push(+v.toFixed(10));
  return out;
}
const legendOf = (items, T) => h('ul', { class: 'legend', 'aria-hidden': 'true' }, items.map(([name, shape, fill]) =>
  h('li', {}, s('svg', { width: 14, height: 14 }, mark(shape, 7, 7, 4.5, fill, T.line)), name)));

// dots with mean ± std whiskers; one category per x slot, one marker shape per series
export function dotChart({ cats, series, refs = [], theme = 'dark', aria }) {
  const T = THEMES[theme], W = 640, H = 300, m = { l: 54, r: 16, t: 16, b: 50 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const all = []; series.forEach(se => se.pts.forEach(p => { if (p) all.push(p.m - (p.sd || 0), p.m + (p.sd || 0)); }));
  refs.forEach(r => all.push(r.v));
  if (!all.length) return null;
  const tk = ticks(Math.min(...all), Math.max(...all)), lo = tk[0], hi = tk[tk.length - 1];
  const y = v => m.t + ih - ((v - lo) / (hi - lo || 1)) * ih;
  const band = iw / cats.length, ns = series.length, step = Math.min(18, (band * 0.7) / ns);
  const svg = s('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': aria });
  tk.forEach(v => svg.append(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: 'currentColor' }), s('text', { class: 'tick', x: m.l - 6, y: y(v) + 3.5, 'text-anchor': 'end' }, fmt(v, 2))));
  if (lo < 0 && hi > 0) svg.append(s('line', { class: 'zero', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }));
  cats.forEach((c, i) => svg.append(s('text', { class: 'tick', x: m.l + (i + 0.5) * band, y: H - m.b + 18, 'text-anchor': 'middle' }, c)));
  refs.forEach(r => svg.append(s('line', { class: 'ref', x1: m.l, x2: W - m.r, y1: y(r.v), y2: y(r.v), stroke: 'currentColor' }), s('text', { class: 'tick sub', x: W - m.r, y: y(r.v) - 4, 'text-anchor': 'end' }, `${r.label} ${fmt(r.v, 3)}`)));
  series.forEach((se, j) => se.pts.forEach((p, i) => {
    if (!p) return;
    const x = m.l + (i + 0.5) * band + (j - (ns - 1) / 2) * step, sd = p.sd || 0;
    const g = s('g', {}, s('title', {}, `${se.name} · ${cats[i]}: ${fmt(p.m, 3)}${sd ? ' ± ' + fmt(sd, 3) : ''}`));
    if (sd) g.append(s('line', { x1: x, x2: x, y1: y(p.m - sd), y2: y(p.m + sd), stroke: T.line === '#0A0614' ? '#F3EEFB' : T.line, 'stroke-width': 1.4 }),
      s('line', { x1: x - 3, x2: x + 3, y1: y(p.m - sd), y2: y(p.m - sd), stroke: T.line === '#0A0614' ? '#F3EEFB' : T.line }),
      s('line', { x1: x - 3, x2: x + 3, y1: y(p.m + sd), y2: y(p.m + sd), stroke: T.line === '#0A0614' ? '#F3EEFB' : T.line }));
    g.append(mark(SHAPES[j % 5], x, y(p.m), 4.5, T.cols[j % 5], T.line)); svg.append(g);
  }));
  return { svg, legend: series.length > 1 ? legendOf(series.map((se, j) => [se.name, SHAPES[j % 5], T.cols[j % 5]]), T) : null };
}

// one row per class: value dot and CI whisker, arid classes as gold squares
export function ciChart({ rows, theme = 'paper', aria }) {
  const T = THEMES[theme], rh = 20, W = 640, m = { l: 150, r: 20, t: 14, b: 40 }, H = m.t + m.b + rows.length * rh;
  const vals = rows.flatMap(r => [r.v, r.lo, r.hi, 0]).filter(Number.isFinite);
  if (!rows.length) return null;
  const tk = ticks(Math.min(...vals), Math.max(...vals)), lo = tk[0], hi = tk[tk.length - 1];
  const x = v => m.l + ((v - lo) / (hi - lo || 1)) * (W - m.l - m.r);
  const svg = s('svg', { class: 'chart', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': aria });
  tk.forEach(v => svg.append(s('line', { class: 'grid', x1: x(v), x2: x(v), y1: m.t, y2: H - m.b, stroke: 'currentColor' }), s('text', { class: 'tick', x: x(v), y: H - m.b + 16, 'text-anchor': 'middle' }, fmt(v, 2))));
  svg.append(s('line', { class: 'zero', x1: x(0), x2: x(0), y1: m.t, y2: H - m.b }));
  rows.forEach((r, i) => {
    const yy = m.t + i * rh + rh / 2, g = s('g', {}, s('title', {}, `${r.label} (${r.arid ? 'arid' : 'other'}): ${fmt(r.v, 3)}, 95% CI ${fmt(r.lo, 3)} to ${fmt(r.hi, 3)}`));
    g.append(s('text', { class: 'gl', x: m.l - 8, y: yy + 3.5, 'text-anchor': 'end', style: r.arid ? 'font-weight:800' : '' }, r.label));
    if (Number.isFinite(r.lo) && Number.isFinite(r.hi)) g.append(s('line', { x1: x(r.lo), x2: x(r.hi), y1: yy, y2: yy, stroke: T.line, 'stroke-width': 1.5 }), s('line', { x1: x(r.lo), x2: x(r.lo), y1: yy - 3, y2: yy + 3, stroke: T.line }), s('line', { x1: x(r.hi), x2: x(r.hi), y1: yy - 3, y2: yy + 3, stroke: T.line }));
    g.append(mark(r.arid ? 'square' : 'circle', x(r.v), yy, 4.5, r.arid ? '#F2C14E' : T.cols[0], T.line)); svg.append(g);
  });
  return { svg, legend: legendOf([['arid class', 'square', '#F2C14E'], ['other class', 'circle', T.cols[0]]], T) };
}

export function tbl(head, body, { hl = () => false, caption = '' } = {}) {
  return h('table', { class: 'tbl' }, caption && h('caption', { class: 'vh' }, caption),
    h('thead', {}, h('tr', {}, head.map(x => h('th', { scope: 'col' }, x)))),
    h('tbody', {}, body.map((r, i) => h('tr', { class: hl(i) ? 'hl' : null }, r.map(c => h('td', { class: typeof c === 'string' && /^[−+-]?\d/.test(c) ? 'num' : null }, c))))));
}
export const rowsTbl = (rows, max = 40) => {
  const keys = Object.keys(rows[0] ?? {});
  return tbl(keys, rows.slice(0, max).map(r => keys.map(k => (typeof r[k] === 'number' ? fmt(r[k], 3) : String(r[k])))));
};

// the table is always in the DOM (visually hidden while the chart shows) so screen readers get a text alternative
export function figure({ title, chart, legend, table }) {
  const wrap = h('div', { class: 'chart-wrap' }, chart), tw = h('div', { class: 'tbl-wrap scroll vh' }, table);
  const b1 = h('button', { type: 'button', class: 'cap ghost tiny', 'aria-pressed': 'true' }, 'Chart');
  const b2 = h('button', { type: 'button', class: 'cap ghost tiny', 'aria-pressed': 'false' }, 'Table');
  const set = t => { b1.setAttribute('aria-pressed', !t); b2.setAttribute('aria-pressed', t); wrap.hidden = t; if (legend) legend.hidden = t; tw.classList.toggle('vh', !t); };
  b1.onclick = () => set(false); b2.onclick = () => set(true);
  return h('figure', { class: 'fig' }, h('figcaption', { class: 'fig-cap' }, title), h('div', { class: 'fig-ctl' }, b1, b2), legend, wrap, tw);
}

export function kvDump(o, max = 120) {
  const dl = h('dl', { class: 'kv' }); let n = 0, extra = 0;
  const walk = (x, p) => {
    if (x && typeof x === 'object') return Object.entries(x).forEach(([k, v]) => walk(v, p ? `${p}.${k}` : k));
    if (n++ >= max) { extra++; return; }
    dl.append(h('dt', {}, p || 'value'), h('dd', {}, String(x)));
  };
  walk(o, '');
  if (extra) dl.append(h('dt', {}, '…'), h('dd', {}, `${extra} more entries not shown`));
  return dl;
}