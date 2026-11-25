import { $, $$ } from './util.js';
import { initFX } from './fx.js';
import { loadData, loadGradcamPaths } from './data.js';
import { initStory } from './story.js';
import { initEvidence } from './evidence.js';
import { initInstrument } from './instrument.js';
import { initMethod } from './method.js';

const safe = (n, f) => { try { return f(); } catch (e) { console.error(n, e); } };

function initDock() {
  const order = ['hook', 'instrument', 'study'];
  const links = Object.fromEntries($$('.dock-list a').map(a => [a.dataset.sec, a])), bars = $$('.glyph i');
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const k = e.target.id; Object.values(links).forEach(a => a.removeAttribute('aria-current')); links[k]?.setAttribute('aria-current', 'true');
    const lit = 1 + Math.floor((order.indexOf(k) * 3) / order.length); bars.forEach((b, j) => b.classList.toggle('lit', j < lit));
  }), { rootMargin: '-45% 0px -50% 0px' });
  order.forEach(id => { const el = document.getElementById(id); if (el) io.observe(el); });
}

function initTabs(onShow) {
  const tabs = $$('[role=tab]'), panels = $$('[role=tabpanel]');
  const show = (id, focus) => {
    tabs.forEach(t => { const on = t.getAttribute('aria-controls') === id; t.setAttribute('aria-selected', on); t.tabIndex = on ? 0 : -1; if (on && focus) t.focus(); });
    panels.forEach(p => { p.hidden = p.id !== id; });
    onShow?.(id);
  };
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => show(t.getAttribute('aria-controls')));
    t.addEventListener('keydown', e => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key]; if (!d) return;
      e.preventDefault(); show(tabs[(i + d + tabs.length) % tabs.length].getAttribute('aria-controls'), true);
    });
  });
}

safe('fx', initFX); safe('dock', initDock);
const [D, GP] = await Promise.all([loadData(), loadGradcamPaths()]);
safe('story', () => initStory(D));
const loadEvidence = safe('evidence', () => initEvidence(D));
safe('instrument', () => initInstrument(D, GP));
safe('method', initMethod);
safe('tabs', () => initTabs(id => { if (id === 'p-results') loadEvidence?.(); }));
