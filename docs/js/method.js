import { CONFIG as C } from './config.js';
import { $, h } from './util.js';
import { loadResultsJSON } from './data.js';
import { kvDump, empty } from './charts.js';

const FILES = ['configs.json', 'tuning.json', 'train_times.json', 'head_val_summary.json', 'audit_a.json', 'audit_b.json', 'features_timing.json', 'uae_coverage.json'];

export function initMethod() {
  const host = $('#todos');
  host.replaceChildren(h('p', { class: 'pix' }, 'Provenance notes'), h('dl', { class: 'kv' }, C.NOTES.flatMap(n => [h('dt', {}, n.label), h('dd', {}, n.text)])));
  host.after(h('div', { class: 'datafiles' }, h('p', { class: 'pix' }, 'Raw exports, rendered as is'), ...FILES.map(f => {
    const body = h('div'); let loaded = false;
    const d = h('details', {}, h('summary', { class: 'mono' }, f), body);
    d.addEventListener('toggle', async () => { if (!d.open || loaded) return; loaded = true; const j = await loadResultsJSON(f); body.replaceChildren(j ? kvDump(j, 160) : empty(f)); });
    return d;
  })));
}