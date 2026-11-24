import { h, REDUCED } from './util.js';
const rnd = (a, b) => a + Math.random() * (b - a);

export function initFX() {
  const sky = document.querySelector('.sky'); if (!sky) return;
  for (let i = 0; i < 26; i++) sky.append(h('i', { class: 'mote', style: `left:${rnd(0, 100).toFixed(1)}%;top:${rnd(0, 100).toFixed(1)}%;--s:${rnd(1.5, 3.5).toFixed(1)}px;--d:${rnd(30, 70).toFixed(0)}s;--dx:${rnd(-60, 60).toFixed(0)}px;--dy:${rnd(-80, 20).toFixed(0)}px;animation-delay:-${rnd(0, 40).toFixed(0)}s` }));
  for (let i = 0; i < 7; i++) sky.append(h('i', { class: 'bubble', style: `left:${rnd(2, 96).toFixed(0)}%;top:${rnd(8, 88).toFixed(0)}%;--b:${rnd(14, 54).toFixed(0)}px;--k:${rnd(.02, .12).toFixed(3)};--bd:${rnd(6, 12).toFixed(1)}s` }));
  if (!REDUCED.matches) {
    let raf = 0;
    addEventListener('scroll', () => { if (raf) return; raf = requestAnimationFrame(() => { raf = 0; document.documentElement.style.setProperty('--sy', scrollY); }); }, { passive: true });
  }
}

// sparkle trail removed (doodles); keep as a no-op so callers still work
export function trail() {}

export function glitter(anchor) {
  if (REDUCED.matches || !anchor) return;
  const r = anchor.getBoundingClientRect(), cols = ['#F2C14E', '#A9E8E8', '#E8A7D1', '#ffffff'];
  for (let i = 0; i < 26; i++) {
    const sz = rnd(2, 6), p = h('i', { class: 'glit', 'aria-hidden': 'true', style: `left:${r.left + r.width * rnd(.05, .95)}px;top:${r.top + r.height * rnd(0, 1)}px;width:${sz}px;height:${sz}px;background:${cols[i % 4]}` });
    document.body.append(p);
    p.animate([{ transform: 'translate(0,0) scale(1)', opacity: 1 }, { transform: `translate(${rnd(-50, 50)}px,${rnd(-100, -30)}px) scale(0)`, opacity: 0 }], { duration: rnd(900, 1600), easing: 'cubic-bezier(.2,.8,.2,1)' }).onfinish = () => p.remove();
  }
}