import { h, s, REDUCED } from './util.js';
const SPR = 'assets/doodles.svg';
const rnd = (a, b) => a + Math.random() * (b - a);

export function initFX() {
  const sky = document.querySelector('.sky'); if (!sky) return;
  const sparkle = (x, y, z) => s('svg', { class: 'spark', width: z, height: z, style: `left:${x}%;top:${y}%;animation-delay:-${rnd(0, 5).toFixed(1)}s`, 'aria-hidden': 'true' }, s('use', { href: `${SPR}#sparkle` }));
  [[8, 14, 18], [22, 62, 10], [47, 9, 14], [71, 28, 22], [88, 12, 12], [93, 55, 16], [35, 38, 9], [60, 70, 12]].forEach(a => sky.append(sparkle(...a)));
  for (let i = 0; i < 26; i++) sky.append(h('i', { class: 'mote', style: `left:${rnd(0, 100).toFixed(1)}%;top:${rnd(0, 100).toFixed(1)}%;--s:${rnd(1.5, 3.5).toFixed(1)}px;--d:${rnd(30, 70).toFixed(0)}s;--dx:${rnd(-60, 60).toFixed(0)}px;--dy:${rnd(-80, 20).toFixed(0)}px;animation-delay:-${rnd(0, 40).toFixed(0)}s` }));
  for (let i = 0; i < 7; i++) sky.append(h('i', { class: 'bubble', style: `left:${rnd(2, 96).toFixed(0)}%;top:${rnd(8, 88).toFixed(0)}%;--b:${rnd(14, 54).toFixed(0)}px;--k:${rnd(.02, .12).toFixed(3)};--bd:${rnd(6, 12).toFixed(1)}s` }));
  if (!REDUCED.matches) {
    let raf = 0;
    addEventListener('scroll', () => { if (raf) return; raf = requestAnimationFrame(() => { raf = 0; document.documentElement.style.setProperty('--sy', scrollY); }); }, { passive: true });
  }
}

// sparkle trail: only while the pointer is over the glass around the imagery, never over the image face
export function trail(el) {
  if (REDUCED.matches) return;
  let last = 0;
  el.addEventListener('pointermove', e => {
    if (e.pointerType === 'touch' || !e.target.closest('.plate') || e.target.closest('.face')) return;
    const t = performance.now(); if (t - last < 55) return; last = t;
    const sp = s('svg', { class: 'trail', width: 14, height: 14, style: `left:${e.clientX - 7}px;top:${e.clientY - 7}px`, 'aria-hidden': 'true' }, s('use', { href: `${SPR}#sparkle` }));
    document.body.append(sp);
    sp.animate([{ opacity: .95, transform: 'scale(.6) rotate(0deg)' }, { opacity: 0, transform: `translate(${rnd(-15, 15)}px,${rnd(10, 34)}px) scale(1.2) rotate(90deg)` }], { duration: 750, easing: 'ease-out' }).onfinish = () => sp.remove();
  });
}

export function glitter(anchor) {
  if (REDUCED.matches || !anchor) return;
  const r = anchor.getBoundingClientRect(), cols = ['#F2C14E', '#A9E8E8', '#E8A7D1', '#ffffff'];
  for (let i = 0; i < 26; i++) {
    const sz = rnd(2, 6), p = h('i', { class: 'glit', 'aria-hidden': 'true', style: `left:${r.left + r.width * rnd(.05, .95)}px;top:${r.top + r.height * rnd(0, 1)}px;width:${sz}px;height:${sz}px;background:${cols[i % 4]}` });
    document.body.append(p);
    p.animate([{ transform: 'translate(0,0) scale(1)', opacity: 1 }, { transform: `translate(${rnd(-50, 50)}px,${rnd(-100, -30)}px) scale(0)`, opacity: 0 }], { duration: rnd(900, 1600), easing: 'cubic-bezier(.2,.8,.2,1)' }).onfinish = () => p.remove();
  }
}