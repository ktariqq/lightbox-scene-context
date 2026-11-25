import { CONFIG as C, FOOTPRINT, fmtM, maskOf } from './config.js';
import { $, $$, h, emit, UserError } from './util.js';
import { PlateStack, bindControls } from './plate-stack.js';
import { decode, buildViews, viewsFromItem } from './preprocess.js';
import { ensureModel, extract, runHead, runParity, disposeFeat, E } from './inference.js';
import { liveCam, camCanvas } from './gradcam.js';
import { exportURL } from './data.js';
import { glitter } from './fx.js';
import { empty } from './charts.js';

const SUBL = { c: 'c', s: 's', g: 'g', cs: 'c+s', cg: 'c+g', sg: 's+g', csg: 'c+s+g' };
const argmax = a => a.reduce((b, x, i) => (x > a[b] ? i : b), 0);
const entropyNorm = p => -p.reduce((a, x) => a + (x > 0 ? x * Math.log(x) : 0), 0) / Math.log(p.length);

export function initInstrument(D, GP) {
  const classes = D?.classes ?? [], arid = new Set(D?.groups?.arid ?? []), items = D?.gallery?.items ?? [];
  const grp = c => (arid.has(c) ? 'arid' : 'other');
  const S = { token: 0, rseq: 0, feat: null, item: null, label: '', views: null, mask: [1, 1, 1], res: null, info: null, cams: null, camNote: '', visible: false, started: false };
  const cache = new Map();
  const stack = new PlateStack($('#ins-stack'), { trail: true });
  bindControls(stack, $('#ins-seg'), $('#ins-explode'));

  const status = t => { $('#ins-status').textContent = t; };
  const progress = (f, label) => { $('#ro-load').hidden = false; $('#ro-prog').value = f; $('#ro-prog-l').textContent = `${label} Â· ${Math.round(f * 100)}%`; };

  /* ---------- warnings ---------- */
  let extraWarn = [];
  function renderWarn(extra) {
    if (extra) extraWarn = extra;
    const O = C.OOD, i = S.info, w = [];
    if (i) {
      if (i.aspect > O.aspect) w.push({ text: `This image is not square (${i.w} Ã— ${i.h}), so the model read the central square and left the edges out.` });
      if (i.minSide < O.minSide) w.push({ text: `The shorter side is ${i.minSide} px. The scene was enlarged to ${C.SIDE} px, so the fine texture the model learned from is missing.` });
      if (i.std < O.flatStd) w.push({ text: 'This image is almost uniform in brightness. It may not be an overhead photo.' });
      w.push({ text: 'The model learned from imagery at about 1 m per pixel covering 1.28 km. On anything else, treat the percentages as rough.' });
    }
    if (S.camNote) w.push({ text: S.camNote });
    $('#ins-warn').replaceChildren(...[...w, ...extraWarn].map(x => h('p', { class: x.err ? 'err' : '' }, x.text, x.retry ? h('button', { type: 'button', class: 'cap tiny', onclick: x.retry }, 'Try again') : null)));
  }

  /* ---------- model loading ---------- */
  let loadP = null;
  const showLoading = () => { const el = $('#model-loading'); if (el) el.hidden = false; $('#ins-status').textContent = 'Loading model… please wait'; };
  const hideLoading = () => { const el = $('#model-loading'); if (el) el.hidden = true; };
  const load = () => {
    showLoading();
    return (loadP ??= ensureModel(progress).then(() => { hideLoading(); $('#ro-load').hidden = true; $('#ins-status').textContent = 'Pick a sample or drop an image'; })
      .catch(err => {
        loadP = null; hideLoading(); $('#ro-load').hidden = true; err.shown = true;
        renderWarn([{ err: true, retry: () => analyseCurrent(), text: err instanceof UserError ? err.message : `The model could not be loaded (${err.message}). Check that model/ holds the exported web/ folder and that the page is served over http(s), not opened from disk.` }]);
        throw err;
      }));
  };

  /* ---------- faders ---------- */
  const fad = $('#ro-gates');
  fad.replaceChildren(...C.VIEWS.map(v => h('div', { class: 'fader off' },
    h('div', { class: 'ftrack', role: 'meter', 'aria-label': `${v.label} view gate`, 'aria-valuemin': 0, 'aria-valuemax': 1, 'aria-valuenow': 0, 'aria-valuetext': 'not read yet' }, h('i', { class: 'ffill' }), h('i', { class: 'fthumb' })),
    h('span', { class: 'pix' }, v.label), h('b', { class: 'fval' }, 'â€”'))));
  const setFaders = (g, mask) => $$('.fader', fad).forEach((f, i) => {
    const on = g && mask[i], t = $('.ftrack', f);
    f.classList.toggle('off', !on); f.style.setProperty('--g', on ? g[i] : 0);
    t.setAttribute('aria-valuenow', on ? g[i].toFixed(3) : 0); t.setAttribute('aria-valuetext', on ? g[i].toFixed(3) : g ? 'off' : 'not read yet');
    $('.fval', f).textContent = on ? g[i].toFixed(3) : g ? 'off' : 'â€”';
  });

  /* ---------- subsets (mirror plate ejection) ---------- */
  const subRow = $('#subsets');
  subRow.replaceChildren(...C.SUBSETS.map(sb => h('button', { type: 'button', class: 'cap ghost tiny', 'data-sb': sb, 'aria-pressed': 'false', onclick: () => stack.setMask(maskOf(sb)) }, SUBL[sb])));
  const syncSub = () => $$('button', subRow).forEach(b => b.setAttribute('aria-pressed', maskOf(b.dataset.sb).join() === S.mask.join()));
  stack.onMask = m => { S.mask = m.slice(); syncSub(); refresh(); };
  syncSub();
  const subsetLabel = () => { const k = C.SUBSETS.find(sb => maskOf(sb).join() === S.mask.join()); return k ? SUBL[k] : 'none'; };

  /* ---------- contact sheet ---------- */
  const filter = $('#cs-filter');
  [...new Set(items.map(i => i.class))].sort().forEach(c => filter.append(h('option', { value: c }, c)));
  const markCS = () => $$('.cs-item').forEach(b => b.setAttribute('aria-current', items[+b.dataset.i] === S.item ? 'true' : 'false'));
  function renderCS() {
    const f = filter.value, list = $('#cs');
    if (!items.length) { list.replaceChildren(h('li', {}, empty('data.json gallery'))); return; }
    list.replaceChildren(...items.map((it, idx) => ({ it, idx })).filter(o => !f || o.it.class === f).map(({ it, idx }) => {
      const ok = it.pred === it.class, src = it.views?.global?.image;
      return h('li', {}, h('button', { type: 'button', class: 'cs-item', 'data-i': idx, 'aria-current': 'false', onclick: () => analyse({ item: it }) },
        src ? h('img', { src: exportURL(src), alt: `Global view of a ${it.class} sample`, width: 72, height: 72, loading: 'lazy', decoding: 'async' }) : h('i', { class: 'nopic' }),
        h('span', { class: 'cs-t' }, h('b', {}, it.class), h('span', { class: 'tagrow' }, h('span', { class: 'tag ' + grp(it.class) }, grp(it.class)), h('span', { class: 'tag ' + (ok ? 'ok' : 'bad') }, ok ? 'model right' : 'model wrong')))));
    }));
    markCS();
  }
  filter.addEventListener('change', renderCS); renderCS();

  /* ---------- input views ---------- */
  function renderInputs() {
    const cs = C.VIEWS.map(v => {
      const fp = FOOTPRINT(v), c = h('canvas', { width: C.INPUT, height: C.INPUT, role: 'img', 'aria-label': `${v.label} model input, ${C.INPUT} by ${C.INPUT} pixels, ${fmtM(fp)} of ground` });
      c.getContext('2d').drawImage(S.views[v.id], 0, 0);
      return h('figure', {}, c, h('figcaption', {}, `${v.label} Â· ${C.INPUT} Ã— ${C.INPUT} px Â· ${fmtM(fp)} Â· ${(fp / C.INPUT).toFixed(2)} m/px`));
    });
    const i = S.info;
    $('#ins-inputs').replaceChildren(h('p', { class: 'pix' }, i ? `source ${i.w} Ã— ${i.h} px â†’ centre square ${Math.min(i.w, i.h)} px â†’ ${C.SIDE} px â†’ three cuts â†’ ${C.INPUT} px` : `stored ${C.INPUT} px views from the exported gallery`), h('div', { class: 'inrow' }, cs));
  }

  /* ---------- readout ---------- */
  function renderSlip(r) {
    const order = [...r.probs.keys()].sort((a, b) => r.probs[b] - r.probs[a]), it = S.item;
    const row = (i, k) => h('li', { class: 'row' + (k === 0 ? ' first' : '') }, h('span', { class: 'rk' }, k + 1), h('span', { class: 'cl' }, classes[i]), h('span', { class: 'tag ' + grp(classes[i]) }, grp(classes[i])), h('span', { class: 'pr' }, (r.probs[i] * 100).toFixed(1) + '%'), h('i', { class: 'pb', style: `--p:${r.probs[i]}` }));
    const full = S.mask.every(Boolean), same = it && classes[order[0]] === it.pred;
    $('#ro-slip').replaceChildren(
      h('div', { class: 'slip-hd' }, h('span', { class: 'pix' }, `READING Â· ${S.label}`), h('span', { class: 'sticker' }, subsetLabel())),
      h('ol', { class: 'rows' }, order.slice(0, 3).map(row)),
      h('p', { class: 'fine' }, 'The top pick falls in the ', h('b', {}, grp(classes[order[0]])), ' group.'),
      h('details', {}, h('summary', {}, `See all ${classes.length} scene types`), h('ol', { class: 'rows all' }, order.map(row))),
      it ? h('p', { class: 'fine' }, 'Labelled: ', h('b', {}, it.class), ` (${grp(it.class)}) Â· stored prediction: ${it.pred} (${it.pred === it.class ? 'correct' : 'wrong'})`,
        full ? ` Â· live top-1 ${same ? 'matches' : 'differs from'} the stored prediction${same ? '' : '. Small differences can come from browser vs Python resampling.'}` : '') : null);
  }
  function timing(fresh) {
    $('#ro-time').textContent = `${fresh ? (S.cached ? 'backbone cached' : `backbone ${S.bbMs.toFixed(0)} ms (3 views)`) : 'backbone cached'} Â· head ${S.headMs.toFixed(1)} ms Â· prep ${S.prep.toFixed(0)} ms Â· backend ${E.backend}`;
  }
  async function renderCompare() {
    const my = S.token, steps = [['Center only', [1, 0, 0]], ['+ surround', [1, 1, 0]], ['+ global', [1, 1, 1]]], rs = [];
    for (const [, m] of steps) rs.push(await runHead(S.feat, m));
    if (my !== S.token) return;
    $('#compare').replaceChildren(...rs.map((r, i) => {
      const top = argmax(r.probs), prev = i ? argmax(rs[i - 1].probs) : null, chg = prev !== null && prev !== top, m = steps[i][1];
      return h('div', { class: 'cmp' + (chg ? ' chg' : '') }, h('span', { class: 'pix' }, steps[i][0]), h('b', {}, classes[top]), h('span', {}, `${(r.probs[top] * 100).toFixed(1)}%`),
        r.gates ? h('span', { class: 'fine' }, C.VIEWS.filter((_, k) => m[k]).map(v => `${v.id[0].toUpperCase()} ${r.gates[C.VIEWS.indexOf(v)].toFixed(2)}`).join(' Â· ')) : null, chg ? h('span', { class: 'sticker' }, 'new answer') : null);
    }));
  }

  /* ---------- run ---------- */
  async function refresh({ fresh = false } = {}) {
    const my = S.token, rs = ++S.rseq; if (!S.feat) return;
    if (!S.mask.some(Boolean)) { renderWarn([{ text: 'Every view is switched off. Turn at least one back on.' }]); status('No view selected'); stack.setGates(null); setFaders(null, S.mask); return; }
    const t = performance.now(), r = await runHead(S.feat, S.mask);
    if (my !== S.token || rs !== S.rseq) return;
    S.headMs = performance.now() - t; S.res = r;
    renderSlip(r); setFaders(r.gates, S.mask); stack.setGates(r.gates); timing(fresh);
    if (r.gates) emit('reading', { gates: Array.from(r.gates), mask: S.mask.slice() });
    const warn = [], H = entropyNorm(r.probs), top = Math.max(...r.probs);
    if (!S.item && (H > C.OOD.entropyNorm || top < C.OOD.top1Min)) warn.push({ text: `The model is split across many scene types (spread ${H.toFixed(2)}, top pick ${(top * 100).toFixed(0)}%). The image may look unlike the overhead scenes it learned from.` });
    renderWarn(warn); status(`${S.label} Â· views ${subsetLabel()}`);
    if (fresh) glitter($('#ro-slip'));
    renderCompare(); updateCam();
  }
  async function analyse(input) {
    const my = ++S.token, t0 = performance.now(); S.lastInput = input; extraWarn = []; S.camNote = '';
    try {
      status('Reading the imageâ€¦');
      let views, info = null;
      if (input.file) { const d = await decode(input.file); try { ({ views, info } = buildViews(d.src, d.w, d.h)); } finally { d.src.close?.(); } S.item = null; S.label = input.file.name; }
      else { views = await viewsFromItem(input.item); S.item = input.item; S.label = input.item.filename; }
      if (my !== S.token) return;
      Object.assign(S, { views, info, prep: performance.now() - t0, res: null, cams: null });
      stack.setViews(views); stack.clearFilm(); stack.setGates(null); setFaders(null, S.mask);
      renderInputs(); renderWarn(); markCS();
      if (!E.ready) status('Loading the modelâ€¦');
      await load(); if (my !== S.token) return;
      const key = input.file ? `u:${input.file.name}:${input.file.size}:${input.file.lastModified}` : `g:${input.item.filename}`;
      let feat = cache.get(key); S.cached = !!feat;
      if (!feat) {
        status('Reading the three viewsâ€¦â€¦'); feat = await extract(views);
        if (my !== S.token) { disposeFeat(feat); return; }
        cache.set(key, feat); S.bbMs = feat.ms;
        for (const [k, f] of cache) { if (cache.size <= 6) break; if (f !== feat) { disposeFeat(f); cache.delete(k); } }
      }
      S.feat = feat; await refresh({ fresh: true });
    } catch (err) {
      if (my !== S.token || err.shown) return;
      renderWarn([{ err: true, text: err instanceof UserError ? err.message : `This image could not be read (${err.message}).` }]); status('Could not read the image');
    }
  }
  const analyseCurrent = () => (S.lastInput ? analyse(S.lastInput) : load().catch(() => {}));

  /* ---------- Grad-CAM ---------- */
  const camSrc = t => { $('#cam-src').textContent = t; };
  const paintCams = () => C.VIEWS.forEach((v, i) => stack.setFilm(v.id, S.cams?.[i] ? { canvas: camCanvas(S.cams[i], $('#cam-cb').checked ? 'cb' : 'gold') } : null));
  const usePre = (p, label) => { C.VIEWS.forEach((v, i) => stack.setFilm(v.id, S.mask[i] && p[v.id] ? { url: exportURL(p[v.id]) } : null)); camSrc(label); };
  async function updateCam() {
    const my = S.token; stack.setFilmOpacity($('#cam-op').value / 100);
    if (!S.feat || !S.res) return;
    if (!$('#cam-on').checked) { stack.clearFilm(); camSrc('off'); return; }
    const sel = $('#cam-class').value, idx = sel === '' ? argmax(S.res.probs) : +sel, pre = S.item && GP?.[S.item.filename];
    const wantLive = !S.item || $('#cam-live').checked || sel !== '';
    S.camNote = '';
    if (!wantLive && pre) { usePre(pre, 'precomputed Â· export colour map'); renderWarn(); return; }
    if (S.feat.conv) {
      try { const cams = await liveCam(S.feat, S.mask, idx); if (my !== S.token) return; S.cams = cams; paintCams(); camSrc(`live Â· ${classes[idx]}`); renderWarn(); return; }
      catch (e) { console.warn('live Grad-CAM failed', e); S.camNote = `Live Grad-CAM failed in this browser (${e.message}).`; }
    } else S.camNote = 'Live Grad-CAM needs a backbone export with the 7Ã—7Ã—1280 conv map and the pooled feature. Gallery images still show precomputed overlays.';
    if (pre) usePre(pre, 'precomputed Â· ignores your class and views'); else { stack.clearFilm(); camSrc('unavailable'); }
    renderWarn();
  }
  $('#cam-class').append(...classes.map((c, i) => h('option', { value: i }, c)));
  $('#cam-on').onchange = updateCam; $('#cam-class').onchange = updateCam; $('#cam-live').onchange = updateCam;
  $('#cam-op').oninput = () => stack.setFilmOpacity($('#cam-op').value / 100);
  $('#cam-cb').onchange = () => { if (S.cams && !S.item || S.cams) paintCams(); };

  /* ---------- airlock: drop, pick, paste ---------- */
  const drop = $('#drop'), fileIn = $('#file'), pick = f => (f ? analyse({ file: f }) : renderWarn([{ text: 'Drop an image file. Links and text cannot be read.' }]));
  fileIn.addEventListener('change', () => { pick(fileIn.files[0]); fileIn.value = ''; });
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileIn.click(); } });
  ['dragenter', 'dragover'].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => pick([...e.dataTransfer.files][0]));
  addEventListener('dragover', e => e.preventDefault()); addEventListener('drop', e => e.preventDefault());
  addEventListener('paste', e => {
    if (e.target.closest?.('input,textarea,select')) return;
    const f = [...(e.clipboardData?.files ?? [])].find(x => x.type.startsWith('image/'));
    if (f) { e.preventDefault(); analyse({ file: f }); } else if (S.visible) renderWarn([{ text: 'The clipboard holds no image. Copy the image itself and paste again.' }]);
  });

  /* ---------- developer drawer ---------- */
  $('#parity-run').onclick = async () => {
    const out = $('#parity-out'); out.textContent = 'Loading modelâ€¦';
    try { await load(); out.textContent = (await runParity()).join('\n'); } catch (e) { out.textContent = e.shown ? 'Model failed to load. See the message above.' : e.message; }
  };

  /* ---------- lazy start when the instrument opens ---------- */
  new IntersectionObserver(es => {
    S.visible = es[0].isIntersecting; if (!S.visible || S.started) return; S.started = true;
    if (items[0]) analyse({ item: items[0] }); else load().catch(() => {});
  }, { threshold: 0.12 }).observe($('#instrument'));
}