import { CONFIG as C, FOOTPRINT, fmtM } from './config.js';
import { $, $$, h, REDUCED, on, fmt, norm, modelLabel } from './util.js';
import { PlateStack, bindControls } from './plate-stack.js';
import { viewsFromItem } from './preprocess.js';
import { loadCSV } from './data.js';
import { empty, tbl } from './charts.js';

export async function initStory(D) {
  const stack = new PlateStack($('#story-stack'), { trail: true });
  bindControls(stack, $('.stage-ctl .seg'), $('#story-explode'));
  chips(D); scaleBar(); specs(); diagram(stack);
  const it = D?.gallery?.items?.[0];
  if (!it) { preview(null); return; }
  try { const v = await viewsFromItem(it); stack.setViews(v); $('#stage-sample').textContent = it.class; preview(v, it); }
  catch { preview(null); }
}

function chips(D) {
  const ul = $('#arid-chips'), a = D?.groups?.arid ?? [];
  ul.replaceChildren(...(a.length ? a.map(c => h('li', {}, c)) : [h('li', {}, 'class list not exported yet')]));
}

function scaleBar() {
  $('#scale-bar').replaceChildren(...C.VIEWS.map(v => {
    const w = ((v.box[2] - v.box[0]) / C.SIDE) * 100;
    return h('div', { class: 'sb ' + v.id[0] }, h('span', { class: 'pix' }, `${v.label} · ${fmtM(FOOTPRINT(v))} wide · ${w.toFixed(0)}% of the scene's width`), h('i', { class: 'bar', style: `--w:${w}%` }));
  }));
}

function preview(v, it) {
  $('#preview-src').textContent = it ? `· a ${it.class} sample` : '· no sample available yet';
  $('#in-preview').replaceChildren(...C.VIEWS.map(vw => {
    const fp = FOOTPRINT(vw), cap = h('figcaption', {}, `${vw.label}: ${C.INPUT} × ${C.INPUT} px, ${fmtM(fp)} of ground, about ${(fp / C.INPUT).toFixed(2)} m per pixel`);
    if (!v) return h('figure', {}, h('div', { class: 'ph', role: 'img', 'aria-label': 'No sample loaded' }), cap);
    const c = h('canvas', { width: C.INPUT, height: C.INPUT, role: 'img', 'aria-label': `${vw.label} view of a ${it.class} sample at ${C.INPUT} by ${C.INPUT} pixels` });
    c.getContext('2d').drawImage(v[vw.id], 0, 0); return h('figure', {}, c, cap);
  }));
}

async function specs() {
  const host = $('#specs'), rows = await loadCSV('efficiency.csv');
  if (!rows) { host.replaceChildren(empty('efficiency.csv')); return; }
  const r = rows.find(x => norm(x.model) === C.PROPOSED) ?? rows[0];
  const int = v => Number(v).toLocaleString('en'), one = u => v => `${Number(v).toFixed(2)} ${u}`;
  const items = [['Trainable parameters', 'trainable_params', int], ['Total parameters', 'total_params', int], ['Backbone parameters', 'backbone_params', int], ['File size', 'size_mb', one('MB')],
    ['Head latency', 'latency_head_ms', one('ms')], ['End-to-end latency', 'latency_e2e_ms', one('ms')], ['Mean training time', 'train_seconds_mean', one('s')]].filter(i => typeof r[i[1]] === 'number');
  const keys = Object.keys(rows[0]).filter(k => k !== 'model');
  host.replaceChildren(h('p', { class: 'fine' }, `Shown for ${modelLabel(r.model)}. Latencies were measured by the export pipeline, so your browser will differ.`),
    h('dl', { class: 'spec' }, items.map(([t, k, f]) => h('div', {}, h('dt', {}, t), h('dd', {}, f(r[k]))))),
    h('details', { class: 'more' }, h('summary', {}, 'All models'), h('div', { class: 'scroll' }, tbl(['model', ...keys], rows.map(x => [modelLabel(x.model), ...keys.map(k => (typeof x[k] === 'number' ? fmt(x[k], 2) : String(x[k])))])))));
}

function diagram(stack) {
  const dg = $('#diagram'), st = { mask: [1, 1, 1], drop: [0, 0, 0] }, ids = C.VIEWS.map(v => v.id);
  const paint = () => $$('.dg-v', dg).forEach(g => { const i = ids.indexOf(g.dataset.v); g.classList.toggle('dark', !st.mask[i] || !!st.drop[i]); });
  stack.onMask = m => { st.mask = m.slice(); paint(); };
  // illustration only: the real dropout schedule is in the exported configs.json
  const roll = () => { const d = [0, 1, 2].map(() => (Math.random() < 0.45 ? 1 : 0)); if (d.every(Boolean)) d[Math.floor(Math.random() * 3)] = 0; st.drop = d; stack.setDark(d); paint(); };
  $('#dropout-btn').addEventListener('click', roll);
  let timer = 0;
  new IntersectionObserver(e => {
    clearInterval(timer);
    if (e[0].isIntersecting) { if (!REDUCED.matches) timer = setInterval(roll, 1800); }
    else { st.drop = [0, 0, 0]; stack.setDark([0, 0, 0]); paint(); }
  }, { threshold: 0.4 }).observe(dg);
  on('reading', ({ gates, mask }) => {
    if (!gates) return;
    $$('[data-fader]', dg).forEach(t => {
      const i = +t.dataset.fader, g = mask[i] ? gates[i] : null;
      t.classList.toggle('live', g != null); t.setAttribute('y', g == null ? 331 : (376 - g * 80 - 4.5).toFixed(1));
      $(`[data-fv="${i}"]`, dg).textContent = g == null ? 'off' : `gate ${g.toFixed(2)}`;
    });
    $('#dg-cap').textContent = 'The faders follow your latest reading. Gates are learned weights and do not prove cause. The random switching shown here is an illustration.';
  });
}
