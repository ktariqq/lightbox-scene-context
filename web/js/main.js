import { $, $$ } from './util.js';
import { initFX } from './fx.js';
import { loadData, loadGradcamPaths } from './data.js';
import { initStory } from './story.js';
import { initEvidence } from './evidence.js';
import { initInstrument } from './instrument.js';
import { initMethod } from './method.js';

const safe = (n, f) => { try { return f(); } catch (e) { console.error(n, e); } };

function initDock() {
  const map = { hook: 'hook', questions: 'questions', q2: 'questions', q3: 'questions', views: 'views', model: 'model', evidence: 'evidence', instrument: 'instrument', method: 'method' };
  const order = ['hook', 'questions', 'views', 'model', 'evidence', 'instrument', 'method'];
  const links = Object.fromEntries($$('.dock-list a').map(a => [a.dataset.sec, a])), bars = $$('.glyph i');
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const k = map[e.target.id]; Object.values(links).forEach(a => a.removeAttribute('aria-current')); links[k]?.setAttribute('aria-current', 'true');
    const lit = 1 + Math.floor((order.indexOf(k) * 3) / order.length); bars.forEach((b, j) => b.classList.toggle('lit', j < lit));
  }), { rootMargin: '-45% 0px -50% 0px' });
  Object.keys(map).forEach(id => { const el = document.getElementById(id); if (el) io.observe(el); });
}

safe('fx', initFX); safe('dock', initDock);
const [D, GP] = await Promise.all([loadData(), loadGradcamPaths()]);
safe('story', () => initStory(D)); safe('evidence', () => initEvidence(D)); safe('instrument', () => initInstrument(D, GP)); safe('method', initMethod);