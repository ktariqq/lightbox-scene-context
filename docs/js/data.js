import { CONFIG as C } from './config.js';
import { resolve } from './util.js';

const cache = new Map();
const get = (url, kind) => {
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then(async r => {
      if (!r.ok) throw new Error(r.status);
      const t = await r.text();
      if (/^\s*</.test(t)) throw new Error('HTML instead of data'); // dev servers that answer 404 with index.html
      return kind === 'json' ? JSON.parse(t) : t;
    }).catch(e => { console.warn('missing', url, e.message); return null; }));
  }
  return cache.get(url);
};

export const exportURL = p => resolve(p, C.EXPORT_ROOT);
export const resultsURL = f => resolve(f, C.RESULTS_ROOT);
export const loadManifest = () => get(exportURL(C.FILES.manifest), 'json');
export const loadData = () => get(exportURL(C.FILES.data), 'json');
export const loadGradcamPaths = () => get(exportURL(C.FILES.gradcam), 'json');
export const loadParity = () => get(exportURL(C.FILES.parity), 'json');
export const loadResultsJSON = f => get(resultsURL(f), 'json');
export const tableFile = (ds, variant, subset) => `table_${ds}_${variant}_${subset}.csv`;

export function parseCSV(text) {
  const rows = []; let row = [], f = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(f); f = ''; if (row.length > 1 || row[0] !== '') rows.push(row); row = []; }
    else f += c;
  }
  if (f !== '' || row.length) { row.push(f); rows.push(row); }
  return rows;
}
const num = v => (v !== undefined && v !== '' && !isNaN(+v) ? +v : v);
export async function loadCSV(file) {
  const t = await get(resultsURL(file), 'text'); if (!t) return null;
  const r = parseCSV(t); const head = r.shift().map(x => x.trim());
  return r.map(a => Object.fromEntries(head.map((k, i) => [k, num(a[i])])));
}
export async function loadRawCSV(file) { const t = await get(resultsURL(file), 'text'); return t ? parseCSV(t) : null; }
export const truthy = v => v === true || /^(true|1|yes)$/i.test(String(v));