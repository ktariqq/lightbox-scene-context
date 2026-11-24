import { CONFIG as C } from './config.js';
import { $, h } from './util.js';
import { loadResultsJSON } from './data.js';
import { kvDump, empty } from './charts.js';

// added by the site, not in your README: the context-gain definition is not stated in any file I was given
const EXTRA = [{ label: 'Context-gain definition', text: '[EDIT: one-line definition of gain_f1 / gain_acc as computed by the export code]' }];
const FILES = ['configs.json', 'tuning.json', 'train_times.json', 'head_val_summary.json', 'audit_a.json', 'audit_b.json', 'features_timing.json', 'uae_coverage.json'];

export function initMethod() {
  const host = $('#todos');
  host.append(h('p', { class: 'pix' }, 'Open placeholders'), ...[...C.TODOS, ...EXTRA].map(t => h('div', {}, h('span', { class: 'pix' }, t.label), ' ', h('span', { class: 'todo' }, t.text))));
  host.after(h('div', { class: 'datafiles' }, h('p', { class: 'pix' }, 'Raw exports, rendered as is'), ...FILES.map(f => {
    const body = h('div'); let loaded = false;
    const d = h('details', {}, h('summary', { class: 'mono' }, f), body);
    d.addEventListener('toggle', async () => { if (!d.open || loaded) return; loaded = true; const j = await loadResultsJSON(f); body.replaceChildren(j ? kvDump(j, 160) : empty(f)); });
    return d;
  })));
}