export const CONFIG = {
  EXPORT_ROOT: '',                // export.py + export_web.py outputs live at the docs/ root
  RESULTS_ROOT: 'data/results/',  // verbatim copy of results/ (copy results/* here)
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
    NOTES: [
    { label: 'MEET licence and citation', text: 'Y. Li, Y. Wu, G. Cheng, C. Tao, B. Dang, Y. Wang, J. Zhang, C. Zhang, Y. Liu, X. Tang, J. Ma, and  Y. Zhang,  “MEET: A million-scale dataset for fine-grained geospatial scene classification with zoom-free remote sensing imagery,” IEEE/CAA J. Autom. Sinica, vol. 12, no. 5, pp. 1004–1023, May 2025. doi: 10.1109/JAS.2025.125324	' },
    { label: 'UAE labelling', text: 'One reviewer (the author) labelled every UAE sample by eye. A second-annotator agreement check was not done.' },
    { label: 'UAE acquisition and coverage', text: 'Sentinel-2 L2A scenes from 2025-11-01 to 2026-03-16 at 10 m per pixel, cloud cover below 1%. 93 reviewed samples in 9 of the 24 classes: Airport 12, Beach 12, Brownfield 7, Landfill 7, Mountain 17, Quarry 6, Scrub 8, Solar power plant 11, Thermal power plant 13.' },
    { label: 'Compute', text: 'Image extraction and feature caching ran on Kaggle with GPU T4. Training, evaluation and the web export ran locally on CPU.' },
    { label: 'Copernicus notice', text: 'Contains modified Copernicus Sentinel data 2025, 2026.' },
    { label: 'Context gain', text: 'For one class: its score with all three views minus its score with the center only (B2), same seed, MEET test set. F1 gain uses per-class F1; accuracy gain uses per-class accuracy (recall). Positive means context helped.' },
  ],
};
export const FOOTPRINT = v => (v.box[2] - v.box[0]) * CONFIG.GSD;
export const maskOf = sb => CONFIG.VIEWS.map(v => (sb.includes(v.id[0]) ? 1 : 0));
export const fmtM = m => (m >= 1000 ? `${m / 1000} km` : `${m} m`);