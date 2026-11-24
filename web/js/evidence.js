import { CONFIG as C } from './config.js';
import { $, h, s, fmt, pm, norm, modelLabel, mean, sd, groupBy } from './util.js';
import { loadCSV, loadRawCSV, loadResultsJSON, truthy, tableFile, resultsURL } from './data.js';
import { empty, figure, dotChart, ciChart, tbl, rowsTbl, sel, chk, kvDump } from './charts.js';

const SHORT = { customcnn: 'B1', b2: 'B2', b3: 'B3', b3vd: 'B3-VD', b4: 'B4', b4vd: 'B4-VD', majority: 'Majority', chance: 'Chance' };
const SUBL = { c: 'c', s: 's', g: 'g', cs: 'c+s', cg: 'c+g', sg: 's+g', csg: 'c+s+g' };
const sh = m => SHORT[norm(m)] ?? String(m);
const mdl = r => norm(r.model);
const panel = (kind, span, ...kids) => h('section', { class: `panel ${kind} span-${span}` }, kids);
const cellOf = (r, k) => (r && typeof r[k + '_mean'] === 'number' ? { m: r[k + '_mean'], sd: r[k + '_std'] || 0 } : null);
const byModel = t => new Map((t ?? []).map(r => [mdl(r), r]));
const signed = v => (typeof v === 'number' ? (v < 0 ? '−' : '+') + Math.abs(v).toFixed(3) : '—');
const todoChip = t => h('span', { class: 'todo' }, t);
const imgFig = (file, alt, cap) => {
  const box = h('div', { class: 'shot' }, h('img', { src: resultsURL(file), alt, loading: 'lazy', onerror: () => box.replaceChildren(empty(file)) }));
  return h('figure', { class: 'fig' }, box, h('figcaption', { class: 'fine' }, cap));
};

export function initEvidence(D) {
  const host = $('#evidence-body'); let done = false;
  const go = () => {
    if (done) return; done = true; host.replaceChildren(
      h('p', { class: 'fine span-12', style: 'grid-column:span 12' }, 'Files without a subset column (per-class, McNemar, group summary, confusion) report the configuration chosen by the export scripts. Check those scripts for the exact view subset.'));
    [ablation, comparisons, gain, uae, calib, gates, confusion, perClass].forEach(f => { try { f(host, D); } catch (e) { console.error(f.name, e); } });
  };
  new IntersectionObserver((e, o) => { if (e[0].isIntersecting) { o.disconnect(); go(); } }, { rootMargin: '700px 0px' }).observe($('#evidence'));
}

/* 1 · seven view subsets */
function ablation(host) {
  const st = { ds: 'test', variant: 'full24', metric: 'f1', models: new Set(['b3', 'b4', 'b4vd']) };
  const fig = h('div'), kpi = h('div', { class: 'kpi' });
  const vsel = sel('Variant', [{ v: 'full24', t: 'full24' }, { v: 'restricted', t: 'restricted' }], st.variant, v => { st.variant = v; update(); });
  vsel.hidden = true;
  const ctl = h('div', { class: 'ctl' },
    sel('Data', [{ v: 'test', t: 'MEET test' }, { v: 'uae', t: 'UAE external (aggregate)' }], st.ds, v => { st.ds = v; vsel.hidden = v !== 'uae'; update(); }), vsel,
    sel('Metric', C.METRICS.map(k => ({ v: k, t: C.METRIC_LABEL[k] })), st.metric, v => { st.metric = v; update(); }),
    ...C.FUSION_ORDER.map(m => chk(sh(m), st.models.has(m), on => { on ? st.models.add(m) : st.models.delete(m); update(); })));
  host.append(
    panel('paper', 8, h('span', { class: 'sticker' }, 'RQ 1'), h('h3', { class: 'h3' }, 'Does context help? Seven view subsets'),
      h('p', {}, 'Same frozen backbone, same head family. Only the views that reach the head change. c = center, s = surround, g = global. Dashed lines are baselines.'), ctl, fig),
    panel('plum up', 4, h('span', { class: 'tape' }, 'difference of means'), kpi,
      h('p', { class: 'fine' }, 'B4-VD with all three views minus B4-VD center-only, in the chosen metric. A difference of means, not a significance test. Compare it with the seed spread.')));
  async function update() {
    const variant = st.ds === 'uae' ? st.variant : 'full24';
    const tabs = await Promise.all(C.SUBSETS.map(sb => loadCSV(tableFile(st.ds, variant, sb))));
    if (!tabs.some(Boolean)) { fig.replaceChildren(empty(tableFile(st.ds, variant, 'c'))); kpi.replaceChildren(); return; }
    const maps = tabs.map(byModel), k = st.metric, models = C.FUSION_ORDER.filter(m => st.models.has(m));
    const series = models.map(m => ({ name: sh(m), pts: maps.map(mp => cellOf(mp.get(m), k)) }));
    const refs = ['b2', 'majority'].map(m => ({ label: sh(m), v: cellOf(maps.find(mp => mp.get(m))?.get(m), k)?.m })).filter(r => r.v != null);
    const dc = dotChart({ cats: C.SUBSETS.map(x => SUBL[x]), series, refs, theme: 'paper', aria: `${C.METRIC_LABEL[k]} by view subset for ${models.map(sh).join(', ')}, mean with standard deviation over seeds. Values are in the table view.` });
    const table = tbl(['model', ...C.SUBSETS.map(x => SUBL[x])], models.map(m => [sh(m), ...maps.map(mp => { const c = cellOf(mp.get(m), k); return c ? pm(c.m, c.sd) : '—'; })]), { caption: `${C.METRIC_LABEL[k]} by view subset` });
    fig.replaceChildren(dc ? figure({ title: `${C.METRIC_LABEL[k]} · ${st.ds === 'uae' ? 'UAE ' + variant : 'MEET test'} · mean ± std over seeds`, chart: dc.svg, legend: dc.legend, table }) : empty('no models selected'));
    const full = cellOf(maps[6].get(C.PROPOSED), k), cen = cellOf(maps[0].get(C.PROPOSED), k);
    kpi.replaceChildren(...(full && cen ? [h('b', {}, signed(full.m - cen.m)), h('span', {}, `c+s+g ${pm(full.m, full.sd)}`), h('span', {}, `c only ${pm(cen.m, cen.sd)}`)] : [empty('B4-VD rows')]));
  }
  update();
}

/* 2 · B2 vs B3 vs B4 + McNemar */
function comparisons(host) {
  const a = h('div'), b = h('div');
  host.append(
    panel('plum', 5, h('h3', { class: 'h3' }, 'B2 vs B3 vs B4'),
      h('p', {}, 'B2 reads the center view only. B3 and B4 fuse all three. That is the context comparison. B1 is a different architecture trained end to end, so B1 vs B2 is not.'), a),
    panel('paper', 7, h('span', { class: 'tape gold' }, 'McNemar · paired predictions'), h('p', {}, 'Exact McNemar on the same test images, per seed, with effect sizes. Rows with the “significant” flag are tinted. Verdicts are printed exactly as exported. Check mcnemar.csv’s script for the sign convention of Δ.'), b));
  (async () => {
    const t = await loadCSV(tableFile('test', 'full24', 'csg'));
    if (!t) a.replaceChildren(empty(tableFile('test', 'full24', 'csg')));
    else {
      const mp = byModel(t), ms = ['b2', ...C.FUSION_ORDER];
      const series = ['acc', 'f1'].map(k => ({ name: C.METRIC_LABEL[k], pts: ms.map(m => cellOf(mp.get(m), k)) }));
      const dc = dotChart({ cats: ms.map(sh), series, theme: 'dark', aria: 'Accuracy and F1 on the MEET test set with all three views, for B2, B3, B3-VD, B4 and B4-VD, mean with standard deviation over seeds. Values are in the table view.' });
      a.replaceChildren(dc ? figure({ title: 'MEET test · c+s+g · mean ± std', chart: dc.svg, legend: dc.legend, table: tbl(['model', 'acc', 'f1', 'n_seeds'], ms.map(m => { const r = mp.get(m); return [sh(m), pm(r?.acc_mean, r?.acc_std), pm(r?.f1_mean, r?.f1_std), String(r?.n_seeds ?? '—')]; })) }) : empty('table values'));
    }
    const rows = await loadCSV('mcnemar.csv');
    if (!rows) { b.replaceChildren(empty('mcnemar.csv')); return; }
    const pairs = groupBy(rows, r => `${norm(r.a)}|${norm(r.b)}`), known = C.COMPARISONS.map(([x, y]) => `${x}|${y}`);
    const keys = [...known.filter(k => pairs.has(k)), ...[...pairs.keys()].filter(k => !known.includes(k))];
    const seeds = [...new Set(rows.map(r => r.seed))];
    const dc = dotChart({ cats: keys.map(k => { const r = pairs.get(k)[0]; return `${sh(r.a)} vs ${sh(r.b)}`; }),
      series: seeds.map(sd_ => ({ name: `seed ${sd_}`, pts: keys.map(k => { const r = pairs.get(k).find(x => x.seed === sd_); return r && typeof r.delta_acc === 'number' ? { m: r.delta_acc, sd: 0 } : null; }) })),
      refs: [{ label: 'no difference', v: 0 }], theme: 'paper', aria: 'Accuracy difference for each compared pair of models, one marker per seed. Values are in the tables below.' });
    const p4 = p => (typeof p === 'number' ? (p < 0.001 ? p.toExponential(1) : fmt(p, 4)) : '—');
    b.replaceChildren(dc ? figure({ title: 'Δ accuracy per seed, as exported', chart: dc.svg, legend: dc.legend, table: h('span', {}, 'See the per-comparison tables below.') }) : '',
      ...keys.map(k => {
        const rs = pairs.get(k), d = rs.map(r => r.delta_acc).filter(Number.isFinite);
        return h('div', { class: 'pair' }, h('h4', { class: 'pix' }, `${modelLabel(rs[0].a)} vs ${modelLabel(rs[0].b)}`),
          h('p', { class: 'fine' }, `Δ accuracy ${pm(mean(d), sd(d))} over ${d.length} seeds`),
          h('div', { class: 'scroll' }, tbl(['seed', 'a only', 'b only', 'acc a', 'acc b', 'Δ acc', 'odds ratio', 'p', 'signif.', 'practical', 'verdict'],
            rs.map(r => [String(r.seed), String(r.a_only_correct), String(r.b_only_correct), fmt(r.acc_a), fmt(r.acc_b), signed(r.delta_acc), fmt(r.odds_ratio, 2), p4(r.p_value), truthy(r.significant) ? 'yes' : 'no', truthy(r.practical) ? 'yes' : 'no', h('span', { class: 'verdict' }, String(r.verdict ?? '—'))]),
            { hl: i => truthy(rs[i].significant), caption: `McNemar results for ${modelLabel(rs[0].a)} vs ${modelLabel(rs[0].b)}` })));
      }));
  })();
}

/* 3 · arid vs other gain */
function gain(host) {
  const left = h('div'), right = h('div');
  host.append(panel('paper', 7, h('span', { class: 'sticker pk' }, 'RQ 2'), h('h3', { class: 'h3' }, 'Is context worth more for arid classes?'), left), panel('plum up', 5, right));
  (async () => {
    const [cg, cgg, gsum] = await Promise.all([loadCSV('context_gain.csv'), loadCSV('context_gain_groups.csv'), loadCSV('group_summary.csv')]);
    if (!cg) { left.replaceChildren(empty('context_gain.csv')); return; }
    const models = [...new Set(cg.map(mdl))], seeds = [...new Set(cg.map(r => r.seed))];
    const st = { model: models.includes(C.PROPOSED) ? C.PROPOSED : models[0], seed: seeds.includes(42) ? 42 : seeds[0], metric: 'f1' };
    const fig = h('div');
    left.replaceChildren(
      h('p', {}, 'Per-class gain with a bootstrap 95% CI (test images resampled within each class, paired across models). The arid/other grouping was fixed before results were seen. ', todoChip('[EDIT: one-line definition of context gain as computed by the export code]')),
      h('div', { class: 'ctl' }, sel('Model', models.map(m => ({ v: m, t: modelLabel(m) })), st.model, v => { st.model = v; draw(); }),
        sel('Seed', seeds.map(x => ({ v: x, t: String(x) })), st.seed, v => { st.seed = +v; draw(); }),
        sel('Metric', [{ v: 'f1', t: 'F1' }, { v: 'acc', t: 'Accuracy' }], st.metric, v => { st.metric = v; draw(); })), fig);
    function draw() {
      const rows = cg.filter(r => mdl(r) === st.model && r.seed === st.seed).map(r => ({ label: String(r.class), v: r['gain_' + st.metric], lo: r[st.metric + '_lo'], hi: r[st.metric + '_hi'], arid: String(r.group).toLowerCase() === 'arid' })).filter(r => Number.isFinite(r.v)).sort((p, q) => q.v - p.v);
      const dc = ciChart({ rows, theme: 'paper', aria: `Context gain in ${st.metric} for each class with a 95% bootstrap interval, arid classes as gold squares. Values are in the table view.` });
      fig.replaceChildren(dc ? figure({ title: `Context gain (${st.metric}) · ${modelLabel(st.model)} · seed ${st.seed}`, chart: dc.svg, legend: dc.legend,
        table: tbl(['class', 'group', 'gain', 'CI low', 'CI high'], rows.map(r => [r.label, r.arid ? 'arid' : 'other', signed(r.v), fmt(r.lo), fmt(r.hi)]), { hl: i => rows[i].arid }) }) : empty('context_gain.csv rows for this selection'));
      const g = (cgg ?? []).filter(r => mdl(r) === st.model && String(r.metric).toLowerCase().includes(st.metric)), r0 = g.find(r => r.seed === st.seed);
      const diffs = g.map(r => r.diff_arid_minus_other).filter(Number.isFinite);
      right.replaceChildren(h('span', { class: 'tape' }, 'arid minus other'),
        r0 ? h('div', { class: 'kpi' }, h('b', {}, signed(r0.diff_arid_minus_other)),
          h('span', {}, `95% CI [${fmt(r0.diff_lo)}, ${fmt(r0.diff_hi)}] · the interval ${r0.diff_lo <= 0 && r0.diff_hi >= 0 ? 'includes' : 'excludes'} 0`),
          h('span', {}, `arid gain ${fmt(r0.gain_arid)} · other gain ${fmt(r0.gain_other)}`),
          diffs.length > 1 ? h('span', {}, `across ${diffs.length} seeds: ${pm(mean(diffs), sd(diffs))}`) : null) : empty('context_gain_groups.csv'),
        g.length ? h('div', { class: 'scroll' }, tbl(['seed', 'diff', 'CI low', 'CI high'], g.map(r => [String(r.seed), signed(r.diff_arid_minus_other), fmt(r.diff_lo), fmt(r.diff_hi)]), { caption: 'Arid minus other, per seed' })) : '',
        h('h4', { class: 'pix' }, 'Group F1 · MEET test · as exported'),
        gsum ? h('div', { class: 'scroll' }, tbl(['model', 'group', 'precision', 'recall', 'f1'], gsum.filter(r => r.ds === 'test' && r.variant === 'full24').map(r => [sh(r.model), String(r.group), fmt(r.precision), fmt(r.recall), fmt(r.f1)]), { caption: 'Group summary' })) : empty('group_summary.csv'));
    }
    draw();
  })();
}

/* 4 · UAE external (aggregates only; no imagery, no per-sample rows) */
function uae(host) {
  const fig = h('div'), cov = h('div'), st = { variant: 'full24', metric: 'f1' };
  host.append(panel('plum', 12, h('span', { class: 'sticker' }, 'External test'), h('h3', { class: 'h3' }, 'RQ 3 · a small UAE external set'),
    h('p', {}, 'Esri World Imagery at 1.0 m/pixel, labels checked by eye, no retraining or tuning on UAE. Only classes with at least 5 accepted samples are evaluated. Dry field, Hills and Military training ground are not sampled. Sensors and acquisition years differ from MEET. Read this as a sanity check with wide uncertainty.'),
    h('div', { class: 'ctl' }, sel('Variant', [{ v: 'full24', t: 'full24' }, { v: 'restricted', t: 'restricted' }], st.variant, v => { st.variant = v; draw(); }),
      sel('Metric', C.METRICS.map(k => ({ v: k, t: C.METRIC_LABEL[k] })), st.metric, v => { st.metric = v; draw(); })), fig,
    h('h4', { class: 'pix' }, 'Coverage summary · uae_coverage.json'), cov,
    h('p', {}, todoChip(C.TODOS[2].text), ' ', todoChip(C.TODOS[1].text))));
  async function draw() {
    const [t1, t2] = await Promise.all(['c', 'csg'].map(sb => loadCSV(tableFile('uae', st.variant, sb))));
    if (!t1 && !t2) { fig.replaceChildren(empty(tableFile('uae', st.variant, 'csg'))); return; }
    const m1 = byModel(t1), m2 = byModel(t2), ms = ['b2', ...C.FUSION_ORDER], k = st.metric;
    const dc = dotChart({ cats: ms.map(sh), series: [{ name: 'center only (c)', pts: ms.map(m => cellOf(m1.get(m), k)) }, { name: 'all views (c+s+g)', pts: ms.map(m => cellOf(m2.get(m), k)) }], theme: 'dark', aria: `UAE ${st.variant} ${C.METRIC_LABEL[k]} for each model, center-only and all views, mean with standard deviation over seeds. Values are in the table view.` });
    fig.replaceChildren(dc ? figure({ title: `UAE ${st.variant} · ${C.METRIC_LABEL[k]} · mean ± std`, chart: dc.svg, legend: dc.legend,
      table: tbl(['model', 'c', 'c+s+g', 'n_seeds'], ms.map(m => { const a = cellOf(m1.get(m), k), b = cellOf(m2.get(m), k); return [sh(m), a ? pm(a.m, a.sd) : '—', b ? pm(b.m, b.sd) : '—', String(m2.get(m)?.n_seeds ?? '—')]; })) }) : empty('values'));
  }
  draw();
  loadResultsJSON('uae_coverage.json').then(j => cov.replaceChildren(j ? kvDump(j) : empty('uae_coverage.json')));
}

/* 5 · calibration */
function calib(host) {
  const fig = h('div'), st = { ds: 'test' };
  host.append(panel('paper', 6, h('h3', { class: 'h3' }, 'Calibration'),
    h('p', {}, 'ECE is the expected calibration error, and lower is better. The reliability diagram is the exported MEET-test plot. On UAE or on uploads, confidence is not a calibrated probability.'),
    sel('Data', [{ v: 'test', t: 'MEET test' }, { v: 'uae', t: 'UAE full24' }], st.ds, v => { st.ds = v; draw(); }), fig,
    imgFig('reliability_test.png', 'Reliability diagram on the MEET test set: observed accuracy against predicted confidence in bins. The ECE numbers are in the table above.', 'reliability_test.png · MEET test'),
    imgFig('roc_pr_test.png', 'ROC and precision–recall curves on the MEET test set for the evaluated models. Summary AUC values are in the exported tables.', 'roc_pr_test.png · MEET test')));
  async function draw() {
    const t = await loadCSV(tableFile(st.ds, 'full24', 'csg'));
    if (!t) { fig.replaceChildren(empty(tableFile(st.ds, 'full24', 'csg'))); return; }
    const mp = byModel(t), ms = ['customcnn', 'b2', ...C.FUSION_ORDER].filter(m => mp.has(m));
    const dc = dotChart({ cats: ms.map(sh), series: [{ name: 'ECE (lower is better)', pts: ms.map(m => cellOf(mp.get(m), 'ece')) }], theme: 'paper', aria: 'Expected calibration error per model, mean with standard deviation over seeds. Values are in the table view.' });
    fig.replaceChildren(dc ? figure({ title: `ECE · ${st.ds === 'test' ? 'MEET test' : 'UAE'} · c+s+g where applicable`, chart: dc.svg, legend: null, table: tbl(['model', 'ECE'], ms.map(m => { const c = cellOf(mp.get(m), 'ece'); return [sh(m), c ? pm(c.m, c.sd) : '—']; })) }) : empty('ece columns'));
  }
  draw();
}

/* 6 · gates */
function gates(host) {
  const a = h('div'), b = h('div');
  host.append(panel('plum', 6, h('h3', { class: 'h3' }, 'What the gates did'),
    h('p', {}, 'Gates are the head’s learned weights on each view. A large gate means the head leaned on that view’s features. It does not mean the view is causally important.'), a, b,
    imgFig('gate_dist.png', 'Distribution of the learned view gates on the test set, as exported by the pipeline.', 'gate_dist.png')));
  loadCSV('gates_group.csv').then(r => a.replaceChildren(h('h4', { class: 'pix' }, 'gates_group.csv · arid vs other'), r?.length ? h('div', { class: 'scroll' }, rowsTbl(r)) : empty('gates_group.csv')));
  loadCSV('gates_corr.csv').then(r => b.replaceChildren(h('h4', { class: 'pix' }, 'gates_corr.csv · gate vs correctness'), r?.length ? h('div', { class: 'scroll' }, rowsTbl(r)) : empty('gates_corr.csv')));
}

/* 7 · confusion map */
function parseMatrix(raw, classes) {
  let rows = raw.map(r => r.map(x => x.trim())); const num = x => x !== '' && !isNaN(+x);
  let colNames = null, rowNames = null;
  if (rows[0].some(x => !num(x))) { colNames = rows[0]; rows = rows.slice(1); }
  if (rows.length && !num(rows[0][0])) { rowNames = rows.map(r => r[0]); rows = rows.map(r => r.slice(1)); if (colNames && colNames.length === rows[0].length + 1) colNames = colNames.slice(1); }
  const M = rows.map(r => r.map(Number)), n = M.length;
  if (!n || M.some(r => r.length !== n)) return null;
  const names = rowNames ?? colNames ?? (classes.length === n ? classes : null);
  return names ? { M, names } : null;
}
const hex = x => [1, 3, 5].map(i => parseInt(x.slice(i, i + 2), 16));
const STOPS = [[0, '#1A1025'], [.25, '#32164F'], [.5, '#663399'], [.75, '#F2C14E'], [1, '#FFFFFF']];
function heat(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) if (t <= STOPS[i][0]) { const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i], f = (t - a) / (b - a), A = hex(ca), B = hex(cb); return `rgb(${A.map((v, j) => Math.round(v + (B[j] - v) * f)).join(',')})`; }
  return '#fff';
}
function confusion(host, D) {
  const classes = D?.classes ?? [], arid = new Set(D?.groups?.arid ?? []), body = h('div'), st = { model: C.PROPOSED };
  host.append(panel('plum', 12, h('span', { class: 'sticker' }, 'seed 42'), h('h3', { class: 'h3' }, 'Confusion map · MEET test'),
    h('p', {}, 'Rows are true classes and columns are predicted classes, sorted into arid (gold) then other. Colour is the share of the true class predicted as the column class, on a square-root scale. The diagonal (correct) is outlined, not coloured. Pink lines mark the top confusions. Hover or focus a label to isolate a class.'),
    h('div', { class: 'ctl' }, sel('Model', ['b2', 'b3', 'b3vd', 'b4', 'b4vd', 'customcnn'].map(m => ({ v: m, t: modelLabel(m) })), st.model, v => { st.model = v; draw(); })), body));
  async function draw() {
    const [raw, tc] = await Promise.all([loadRawCSV(`confusion_test_${st.model}.csv`), loadCSV('top_confusions.csv')]);
    const mat = raw && parseMatrix(raw, classes);
    if (!mat) { body.replaceChildren(empty(`confusion_test_${st.model}.csv`)); return; }
    const { M, names } = mat, n = names.length, cs = 22, ML = 150, MT = 118, W = ML + n * cs + 80, H = MT + n * cs + 12;
    const ord = [...names.keys()].sort((p, q) => (arid.has(names[q]) - arid.has(names[p])) || names[p].localeCompare(names[q]));
    const pos = new Map(ord.map((o, p) => [names[o], p])), nA = ord.filter(o => arid.has(names[o])).length;
    const svg = s('svg', { class: 'cm', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Confusion matrix for ${modelLabel(st.model)} on the MEET test set, ${n} classes, rows true, columns predicted, sorted into arid then other. The top confusions are listed in the table view.` });
    ord.forEach((ri, r) => {
      const sum = M[ri].reduce((x, y) => x + y, 0);
      ord.forEach((ci, c) => {
        const cnt = M[ri][ci], rate = sum ? cnt / sum : 0, g = s('g', { class: 'cell', 'data-r': r, 'data-c': c }, s('title', {}, `true ${names[ri]} → predicted ${names[ci]}: ${cnt} (${(rate * 100).toFixed(1)}%)`));
        g.append(r === c ? s('rect', { class: 'diag', x: ML + c * cs + 3, y: MT + r * cs + 3, width: cs - 6, height: cs - 6 }) : s('rect', { x: ML + c * cs, y: MT + r * cs, width: cs - 1, height: cs - 1, fill: heat(Math.sqrt(rate)) }));
        svg.append(g);
      });
      const isA = arid.has(names[ri]);
      svg.append(s('text', { class: 'lab' + (isA ? ' arid' : ''), x: ML - 8, y: MT + r * cs + cs / 2 + 3.5, 'text-anchor': 'end', tabindex: 0, 'data-k': r }, names[ri]));
      if (isA) svg.append(s('rect', { class: 'gold', x: 2, y: MT + r * cs + 2, width: 3, height: cs - 4 }));
      const cx = ML + r * cs + cs / 2;
      svg.append(s('text', { class: 'lab' + (isA ? ' arid' : ''), x: cx, y: MT - 8, transform: `rotate(-55 ${cx} ${MT - 8})`, tabindex: 0, 'data-k': r }, names[ri]));
    });
    if (nA && nA < n) svg.append(s('line', { class: 'sep', x1: 0, x2: ML + n * cs, y1: MT + nA * cs, y2: MT + nA * cs }), s('line', { class: 'sep', x1: ML + nA * cs, x2: ML + nA * cs, y1: MT - 4, y2: MT + n * cs }));
    const top = (tc ?? []).filter(r => r.ds === 'test' && norm(r.model) === norm(st.model)).sort((p, q) => q.count - p.count).slice(0, 8);
    top.forEach(t => {
      const r = pos.get(t.true), c = pos.get(t.pred); if (r == null || c == null) return;
      const x = ML + c * cs + cs / 2, y = MT + r * cs + cs / 2;
      svg.append(s('path', { class: 'conn', 'data-r': r, 'data-c': c, d: `M${ML - 4} ${y} H${x} V${MT - 4}` }), s('rect', { class: 'box', 'data-r': r, 'data-c': c, x: ML + c * cs, y: MT + r * cs, width: cs, height: cs }));
    });
    const iso = k => { svg.classList.toggle('iso', k != null); svg.querySelectorAll('[data-r]').forEach(e => e.classList.toggle('dim', k != null && +e.dataset.r !== k && +e.dataset.c !== k)); };
    svg.addEventListener('pointerover', e => { const t = e.target.closest('[data-r],[data-k]'); if (t) iso(+(t.dataset.k ?? t.dataset.r)); });
    svg.addEventListener('pointerleave', () => iso(null));
    svg.addEventListener('focusin', e => { if (e.target.dataset?.k != null) iso(+e.target.dataset.k); });
    svg.addEventListener('focusout', () => iso(null));
    const table = top.length ? tbl(['true', 'predicted', 'count', 'rate of true class'], top.map(t => [String(t.true), String(t.pred), String(t.count), fmt(t.rate_of_true_class)])) : empty('top_confusions.csv');
    body.replaceChildren(h('div', { class: 'cm-legend' }, '0', h('i'), '100% of the true class'),
      figure({ title: `${modelLabel(st.model)} · seed 42 (as exported)`, chart: h('div', { class: 'scroll' }, svg), legend: null, table }));
  }
  draw();
}

/* 8 · per-class explorer */
function perClass(host) {
  const out = h('div');
  host.append(panel('paper', 12, h('span', { class: 'sticker alt' }, 'MEET test'), h('h3', { class: 'h3' }, 'Per-class explorer'), out));
  (async () => {
    const [pc, tc] = await Promise.all([loadCSV('per_class.csv'), loadCSV('top_confusions.csv')]);
    if (!pc) { out.replaceChildren(empty('per_class.csv')); return; }
    let rows = pc.filter(r => r.ds === 'test' && (r.variant === 'full24' || r.variant == null)); if (!rows.length) rows = pc;
    const models = [...new Set(rows.map(mdl))], st = { model: models.includes(C.PROPOSED) ? C.PROPOSED : models[0], metric: 'f1', sort: 'metric', cls: null };
    const list = h('ul', { class: 'pcl' }), det = h('div', { 'aria-live': 'polite' });
    out.replaceChildren(h('div', { class: 'ctl' }, sel('Model', models.map(m => ({ v: m, t: modelLabel(m) })), st.model, v => { st.model = v; draw(); }),
      sel('Metric', ['f1', 'precision', 'recall'].map(k => ({ v: k, t: k })), st.metric, v => { st.metric = v; draw(); }),
      sel('Sort', [{ v: 'metric', t: 'by score' }, { v: 'name', t: 'by name' }], st.sort, v => { st.sort = v; draw(); })), h('div', { class: 'pc-grid' }, list, det));
    function draw() {
      const per = groupBy(rows.filter(r => mdl(r) === st.model), r => r.class);
      const items = [...per].map(([c, rs]) => ({ c, v: mean(rs.map(r => r[st.metric]).filter(Number.isFinite)), arid: String(rs[0].group).toLowerCase() === 'arid' }));
      items.sort((p, q) => (st.sort === 'name' ? String(p.c).localeCompare(q.c) : q.v - p.v));
      list.replaceChildren(...items.map(it => h('li', { class: it.arid ? 'arid' : '' }, h('button', { type: 'button', 'aria-current': st.cls === it.c ? 'true' : null, onclick: () => { st.cls = it.c; draw(); } },
        h('span', {}, `${it.c}${it.arid ? ' (arid)' : ''}`), h('span', { class: 'pcb' }, h('i', { style: `width:${Math.max(0, Math.min(1, it.v)) * 100}%` })), h('span', {}, fmt(it.v))))));
      if (!st.cls) { det.replaceChildren(h('p', { class: 'fine' }, 'Pick a class to compare models and see its top confusions. Scores are means over seeds.')); return; }
      const g = groupBy(rows.filter(r => r.class === st.cls), mdl);
      const conf = (tc ?? []).filter(r => r.ds === 'test' && norm(r.model) === st.model && r.true === st.cls).sort((p, q) => q.count - p.count).slice(0, 3);
      det.replaceChildren(h('h4', { class: 'h3' }, st.cls),
        h('div', { class: 'scroll' }, tbl(['model', 'precision', 'recall', 'f1', 'support'], [...g].map(([m, rs]) => { const f = k => { const v = rs.map(r => r[k]).filter(Number.isFinite); return pm(mean(v), sd(v)); }; return [sh(m), f('precision'), f('recall'), f('f1'), String(rs[0].support)]; }), { caption: `Per-class scores for ${st.cls}` })),
        h('h4', { class: 'pix' }, `Top confusions · ${modelLabel(st.model)}`),
        conf.length ? h('ul', { class: 'lim' }, conf.map(r => h('li', {}, `predicted as ${r.pred}: ${r.count} (rate of true class ${fmt(r.rate_of_true_class)})`))) : h('p', { class: 'fine' }, 'No confusions listed for this class in top_confusions.csv.'));
    }
    draw();
  })();
}