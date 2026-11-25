# Results files

Written by `evaluate.py`, `explain.py`, `train.py`, `03_audit.py` and `09_s2_summary.py`. Datasets: `test` (MEET test), `uae` (Sentinel-2 external set), `meet10` (MEET test degraded to 10 m).

| File | Contents |
|---|---|
| `runs.csv` | ds, variant, model, seed, subset, acc, top3, precision, recall, f1, roc_auc, pr_auc, ece. Subset is c, s, g, cs, cg, sg, csg for fusion heads and `all` for majority, chance, customcnn, b2. Variant is `full24`; UAE also has `restricted` |
| `table_{test,uae,meet10}_{full24,restricted}_{subset}.csv` | model, `<metric>_mean`, `<metric>_std` for the 8 metrics, n_seeds. Baselines are in every table. `restricted` exists for UAE only |
| `per_class.csv`, `per_class_arid.csv`, `per_class_other.csv` | Per-class precision, recall, f1, support, roc_auc, pr_auc (fusion heads with c+s+g) |
| `group_summary.csv` | Arid vs other means per model |
| `errors.csv` | Misclassified samples with confidence |
| `top_confusions.csv`, `confusion_{ds}_{model}.csv` | Ten largest confusions per model; full matrices (seed 42) |
| `mcnemar.csv` | Exact McNemar per seed on MEET test: a, b, seed, a_only_correct, b_only_correct, acc_a, acc_b, delta_acc (b minus a), odds_ratio, p_value, significant, practical, verdict |
| `context_gain.csv`, `context_gain_groups.csv` | Per-class gain (fusion minus B2, F1 and accuracy) with stratified paired bootstrap 95% CIs; arid-minus-other difference |
| `efficiency.csv`, `features_timing.json` | Parameters, size, CPU latency at batch size 1; feature extraction timing |
| `gates_samples.csv`, `gates_per_class.csv`, `gates_group.csv`, `gates_corr.csv` | Per-sample gates, class and group means, gate-vs-correctness correlation |
| `s2_summary_classes.csv`, `s2_summary_subsets.csv` | Test, MEET-10m and UAE on the same covered classes |
| `curves.json`, `roc_pr_test.png`, `reliability_test.png`, `gate_dist.png` | Curve data and figures |
| `gallery.json`, `gradcam_paths.json`, `gradcam/` | Gallery metadata (3 correct and 2 wrong per class) and Grad-CAM overlays |
| `metrics.json` | Combined summary of the above |
| `train_times.json`, `history.json`, `tuning.json`, `configs.json`, `head_val_summary.json` | Training and tuning records |
| `audit_a.json`, `audit_b.json`, `audit_dup_md5.csv`, `audit_b_top20.csv`, `uae_coverage.json` | Leakage screens and UAE coverage |
| `kt/` (not committed) | Raw keras-tuner trials (`oracle.json`, `tuner0.json`, `trial.json` per trial), summarised in `tuning.json` |
| `gradcam/` (not committed) | Grad-CAM overlays; the gallery copies live in `docs/cam/` |

Context gain for a class is its score with all three views minus its score with the center only (B2), same seed. `gain_acc` is per-class accuracy (recall).
