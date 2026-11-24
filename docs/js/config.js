export const CONFIG = {
  EXPORT_ROOT: 'model/',          // folder holding a verbatim copy of export's web/ directory
  RESULTS_ROOT: 'data/results/',  // verbatim copy of results/
  FILES: { manifest: 'manifest.json', data: 'data.json', gradcam: 'gradcam_paths.json', parity: 'parity.json' },
  TF: {
    core: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
    wasm: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js',
    wasmPaths: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/',
  },
  SIDE: 1280, INPUT: 224, GSD: 1.0,
  // boxes are (left, top, right, bottom) in the 1280 px frame, as in the README
  VIEWS: [
    { id: 'center',   label: 'Center',   box: [512, 512, 768, 768] },
    { id: 'surround', label: 'Surround', box: [256, 256, 1024, 1024] },
    { id: 'global',   label: 'Global',   box: [0, 0, 1280, 1280] },
  ],
  SUBSETS: ['c', 's', 'g', 'cs', 'cg', 'sg', 'csg'],
  METRICS: ['acc', 'top3', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'ece'],
  METRIC_LABEL: { acc: 'Accuracy', top3: 'Top-3 accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1', roc_auc: 'ROC AUC', pr_auc: 'PR AUC', ece: 'ECE (lower is better)' },
  SEEDS: [42, 7, 123],
  PROPOSED: 'b4vd',
  BASELINES: ['majority', 'chance', 'customcnn', 'b2'],
  FUSION_ORDER: ['b3', 'b3vd', 'b4', 'b4vd'],
  COMPARISONS: [['b2', 'b3'], ['b3', 'b4'], ['b3vd', 'b4vd'], ['b3', 'b3vd'], ['b4', 'b4vd']],
  MAX_FILE_MB: 25,
  ACCEPT: ['image/jpeg', 'image/png', 'image/webp'],
  // heuristic thresholds, not tuned on data
  OOD: { aspect: 1.15, minSide: 640, flatStd: 6, entropyNorm: 0.6, top1Min: 0.35 },
  TODOS: [
    { label: 'MEET licence and citation', text: '[EDIT: copy from the Hugging Face dataset card]' },
    { label: 'UAE second-annotator agreement', text: '[EDIT: report kappa and n, or state not done]' },
    { label: 'UAE acquisition and coverage', text: '[EDIT: acquisition dates min-max, source resolution min/median/max, flagged counts, final class coverage from results/uae_coverage.json]' },
    { label: 'Extraction platform', text: '[EDIT: Kaggle or Colab, GPU type]' },
    { label: 'Copernicus attribution', text: '[EDIT: paste copyrightText from uae/service_info.json]' },
  ],
};
export const FOOTPRINT = v => (v.box[2] - v.box[0]) * CONFIG.GSD;
export const maskOf = sb => CONFIG.VIEWS.map(v => (sb.includes(v.id[0]) ? 1 : 0));
export const fmtM = m => (m >= 1000 ? `${m / 1000} km` : `${m} m`);