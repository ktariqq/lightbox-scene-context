import { CONFIG as C, FOOTPRINT, fmtM } from './config.js';
import { h, REDUCED } from './util.js';
import { trail } from './fx.js';

const RATIO = {
  ground: Object.fromEntries(C.VIEWS.map(v => [v.id, (v.box[2] - v.box[0]) / C.SIDE])),
  input: Object.fromEntries(C.VIEWS.map(v => [v.id, 0.6])),
};

export class PlateStack {
  constructor(host, { trail: withTrail = false, onMask = null } = {}) {
    Object.assign(this, { host, onMask, mask: [1, 1, 1], dark: [0, 0, 0], mode: 'ground', explode: 0.4, gates: null, P: 300, gap: 40, tx: 0, ty: 0, cx: 0, cy: 0, visible: true, raf: 0, onMode: null, onExplode: null });
    this.build();
    new ResizeObserver(() => this.resize()).observe(host);
    new IntersectionObserver(e => { this.visible = e[0].isIntersecting; if (this.visible) this.start(); }).observe(host);
    host.addEventListener('pointermove', e => { if (REDUCED.matches) return; const r = host.getBoundingClientRect(); this.tx = ((e.clientX - r.left) / r.width - .5) * 2; this.ty = ((e.clientY - r.top) / r.height - .5) * 2; });
    host.addEventListener('pointerleave', () => { this.tx = this.ty = 0; });
    if (withTrail) trail(host);
    this.initTilt();
    this.resize(); this.setMode(this.mode); this.frame(0); this.start();
  }

  build() {
    const host = this.host; host.textContent = ''; host.classList.add('stack-host');
    this.live = h('p', { class: 'vh', role: 'status', 'aria-live': 'polite' });
    this.rot = h('div', { class: 'stack-rot' });
    this.beams = h('div', { class: 'beams', 'aria-hidden': 'true' });
    this.segs = [0, 1, 2, 3].map(() => { const A = h('i', { class: 'bp' }), B = h('i', { class: 'bp' }); const el = h('div', { class: 'bseg' }, A, B); this.beams.append(el); return { el, A, B }; });
    this.rot.append(this.beams);
    this.plates = C.VIEWS.map((v, i) => {
      const fp = fmtM(FOOTPRINT(v));
      const pic = h('canvas', { class: 'pic', width: C.INPUT, height: C.INPUT, role: 'img', 'aria-label': `${v.label} view, ${fp} wide` });
      const fc = h('canvas', { class: 'film', width: C.INPUT, height: C.INPUT, hidden: true, 'aria-hidden': 'true' });
      const fi = h('img', { class: 'film', alt: '', hidden: true, 'aria-hidden': 'true' });
      const gtag = h('span', { class: 'gtag pix' });
      const el = h('div', { class: 'plate', 'data-view': v.id, role: 'switch', tabindex: 0, 'aria-checked': 'true' },
        h('div', { class: 'face' }, pic, fc, fi, h('i', { class: 'glint' })),
        ['t', 'r', 'b', 'l'].map(k => h('i', { class: 'edge e-' + k })),
        ['a', 'b', 'c', 'd'].map(k => h('i', { class: 'pj pj-' + k })),
        h('span', { class: 'plabel pix' }, `PLATE 0${i + 1}/03 · ${v.label} · ${fp}`), gtag);
      const p = { v, el, pic, fc, fi, gtag, fp, z: 0 };
      let drag = null;
      el.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, done: false }; });
      el.addEventListener('pointermove', e => {
        if (!drag || drag.done || !(e.buttons & 1)) return;
        if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 56 && this.mask[i]) { drag.done = true; this.toggle(i); }
      });
      el.addEventListener('click', () => { if (drag?.done) { drag = null; return; } drag = null; this.toggle(i); });
      el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.toggle(i); } });
      this.rot.append(el);
      return p;
    });
    host.append(this.rot, this.live);
    this.labelAria();
  }

  labelAria() {
    this.plates.forEach((p, i) => {
      const on = !!this.mask[i];
      p.el.setAttribute('aria-checked', on);
      p.el.setAttribute('aria-label', `${p.v.label} view, ${p.fp} wide. ${on ? 'Active. Press Enter to eject it.' : 'Ejected. Press Enter to restore it.'}`);
      p.el.classList.toggle('ejected', !on);
      p.el.classList.toggle('dark', !!this.dark[i]);
    });
  }
  toggle(i) { const m = this.mask.slice(); m[i] = m[i] ? 0 : 1; this.live.textContent = `${this.plates[i].v.label} view ${m[i] ? 'restored' : 'ejected'}`; this.setMask(m); }
  setMask(m, { silent = false } = {}) { this.mask = m.slice(); this.labelAria(); this.updateBeam(); if (!silent && this.onMask) this.onMask(this.mask); }
  setDark(d) { this.dark = d.slice(); this.labelAria(); }
  setFocus(id) { if (id) this.host.dataset.focus = id; else delete this.host.dataset.focus; this.plates.forEach(p => p.el.classList.toggle('is-focus', p.v.id === id)); }
  setMode(m) { this.mode = m; this.host.dataset.mode = m; this.plates.forEach(p => p.el.style.setProperty('--r', RATIO[m][p.v.id])); this.layout(); this.onMode?.(m); }
  setExplode(e) { this.explode = e; this.layout(); this.onExplode?.(e); }
  setGates(g) { this.gates = g ? Array.from(g) : null; this.updateBeam(); }

  setViews(map) {
    this.plates.forEach(p => {
      const src = map?.[p.v.id]; p.el.classList.toggle('empty', !src);
      const x = p.pic.getContext('2d'); x.clearRect(0, 0, C.INPUT, C.INPUT);
      if (src) x.drawImage(src, 0, 0, C.INPUT, C.INPUT);
    });
  }
  setFilm(id, src) {
    const p = this.plates.find(q => q.v.id === id); if (!p) return;
    if (!src) { p.fc.hidden = p.fi.hidden = true; return; }
    if (src.canvas) { const x = p.fc.getContext('2d'); x.clearRect(0, 0, C.INPUT, C.INPUT); x.drawImage(src.canvas, 0, 0); p.fc.hidden = false; p.fi.hidden = true; }
    else if (src.url) { p.fi.onerror = () => { p.fi.hidden = true; }; p.fi.src = src.url; p.fi.hidden = false; p.fc.hidden = true; }
    this.host.classList.remove('bloom'); void this.host.offsetWidth; this.host.classList.add('bloom');
  }
  clearFilm() { this.plates.forEach(p => { p.fc.hidden = true; p.fi.hidden = true; }); }
  setFilmOpacity(o) { this.host.style.setProperty('--film-o', o); }

  resize() {
    const r = this.host.getBoundingClientRect();
    this.P = Math.max(110, Math.min(r.width * 0.5, r.height * 0.62, 400));
    this.host.style.setProperty('--P', this.P + 'px'); this.layout();
  }
  layout() {
    const gap = this.P * 0.04 + this.explode * this.P * 0.42; this.gap = gap;
    this.plates.forEach((p, i) => { p.z = (1 - i) * gap; p.el.style.setProperty('--z', p.z + 'px'); p.el.style.setProperty('--ph', gap + 'px'); });
    this.updateBeam();
  }

  // beam: width and brightness at each plate = that view's gate; ejected plates pass no light
  updateBeam() {
    const P = this.P, g = this.gates, m = this.mask, gap = this.gap, L = gap * 0.9 + P * 0.07;
    const gv = i => (!m[i] ? 0 : g ? Math.max(0, Math.min(1, g[i])) : null);
    const W = i => { const v = gv(i); return v === null ? 5 : v === 0 ? 0 : 5 + v * 0.3 * P; };
    const A = i => { const v = gv(i); return v === null ? 0.16 : v === 0 ? 0 : 0.12 + 0.8 * v; };
    const z = this.plates.map(p => p.z);
    [[z[0] + L, L, 0, 0, W(0), A(0)], [z[0], gap, W(0), A(0), W(1), A(1)], [z[1], gap, W(1), A(1), W(2), A(2)], [z[2], L, W(2), A(2), 0, 0]]
      .forEach((d, i) => this.paint(this.segs[i], ...d));
    this.plates.forEach((p, i) => {
      p.el.style.setProperty('--gl', gv(i) || 0);
      p.gtag.textContent = !m[i] ? 'OFF' : g ? `G ${g[i].toFixed(2)}` : 'G —';
    });
  }
  paint(seg, z, len, w0, a0, w1, a1) {
    const mx = Math.max(w0, w1, 1); seg.el.style.transform = `translateZ(${z}px)`;
    const a = 50 - (w0 / mx) * 50, b = 50 + (w0 / mx) * 50, c = 50 - (w1 / mx) * 50, d = 50 + (w1 / mx) * 50;
    for (const pl of [seg.A, seg.B]) {
      const st = pl.style; st.width = mx + 'px'; st.height = len + 'px'; st.marginLeft = -mx / 2 + 'px';
      st.clipPath = `polygon(${a}% 0,${b}% 0,${d}% 100%,${c}% 100%)`;
      st.background = `linear-gradient(180deg,rgba(255,243,196,${a0}),rgba(255,243,196,${a1}))`;
    }
  }

  start() {
    if (this.raf || REDUCED.matches) return;
    const f = t => { this.raf = 0; if (!this.visible || REDUCED.matches) return; this.frame(t); this.raf = requestAnimationFrame(f); };
    this.raf = requestAnimationFrame(f);
  }
  frame(t) {
    const k = 0.07, d = REDUCED.matches ? 0 : 1;
    this.cx += (this.tx - this.cx) * k; this.cy += (this.ty - this.cy) * k;
    const rx = 48 + this.cy * 8 * d + Math.sin(t / 1700) * 2 * d, rz = -24 + this.cx * 13 * d + Math.cos(t / 2300) * 2.5 * d;
    this.rot.style.transform = `rotateX(${rx}deg) rotateZ(${rz}deg)`;
    this.host.style.setProperty('--gx', (50 + this.cx * 55).toFixed(1) + '%');
  }
  initTilt() {
    const attach = () => addEventListener('deviceorientation', e => {
      if (REDUCED.matches || e.gamma == null) return;
      this.tx = Math.max(-1, Math.min(1, e.gamma / 30)); this.ty = Math.max(-1, Math.min(1, ((e.beta ?? 45) - 45) / 30));
    });
    if (!('DeviceOrientationEvent' in window) || !matchMedia('(pointer:coarse)').matches) return;
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
      const b = h('button', { type: 'button', class: 'cap tiny tilt-btn', onclick: async () => { try { if (await DeviceOrientationEvent.requestPermission() === 'granted') attach(); } catch {} b.remove(); } }, 'Enable tilt');
      this.host.append(b);
    } else attach();
  }
}

export function bindControls(stack, seg, range) {
  const sync = m => seg.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-checked', b.dataset.mode === m));
  seg.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => stack.setMode(b.dataset.mode)));
  range.addEventListener('input', () => stack.setExplode(range.value / 100));
  stack.onMode = sync; stack.onExplode = e => { range.value = Math.round(e * 100); };
  stack.setExplode(range.value / 100);
}