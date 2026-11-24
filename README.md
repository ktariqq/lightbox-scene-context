# AIRE410: Context-aware fine-grained scene classification (MEET subset, 24 classes)

Research questions: (1) does multi-scale context help versus center-only, (2) is the gain larger for arid/bare classes, (3) does it hold on a small UAE external set. Proposed method: gated multi-scale fusion head trained with view dropout (B4-VD).

Placeholders written as `[EDIT: ...]` must be filled before submission.

## Data card: MEET
- Source: Hugging Face dataset repo `zinStar/MEET`. 80 zips named `<Class name>.zip`, plus `MEET_all_1125.txt`, `MEET_train_1125.txt` (620,309 lines), `MEET_valid_1125.txt`, `MEET_test_1125.txt` (206,719 lines each).
- Sample: one 1280x1280 RGB JPEG at 1.0 m/pixel (1.28 km on the ground). Filenames are `<Category>_<id>.jpg`.
- Split line format: `<filename> <class name> <class index 0-79>`.
- No coordinates or location metadata exist, so geographic overlap between splits cannot be checked. Near-duplicate scenes are screened by exact MD5 (`03_audit.py a`) and by feature cosine similarity to the nearest train sample (`03_audit.py b`).
- Class sizes vary greatly (for example Sand.zip holds 1460 images).
- MEET license and citation: [EDIT: copy from the Hugging Face dataset card].

## Subset manifest (`subset_manifest.csv`)
- 24 classes (sorted order = label index, `classes.json`), sampled only from the official split lists, never reshuffled or reassigned.
- Per class and split: min(N, available), N = 200 train / 50 val / 100 test, seed 42 (`config.py`).
- Columns: filename, class, split, md5 (of the original bytes), ok. Counts and availability are in `subset_counts.csv`; extraction log in `extract_log.csv`.
- Views are nested crops resized to 224x224 (bicubic): center box (512,512,768,768), surround (256,256,1024,1024), global = full image.
- Arid vs other grouping: `classes.json["arid"]`, fixed before results were seen.

## UAE external set
- Candidates: OpenStreetMap via Overpass (candidate generation only, tags in `uae/candidates.csv`). Military elements dropped, centers at least 2.5 km apart, up to 25 per class. Dry field, Hills, and Military training ground are not sampled.
- Imagery: Esri World Imagery (current release), `World_Imagery/MapServer/export`, requested at exactly 1.0 m/pixel (1280 m footprint in EPSG:3857 using 640/cos(lat) half-width) to match MEET.
- Ground truth is manual visual review (`06_uae_review.py`); OSM labels are not trusted. Optional second-annotator agreement: `06b_uae_kappa.py` [EDIT: report kappa and n, or state not done].
- Acquisition date range, source resolution, and flags: see `uae_manifest.csv`, `uae/meta.csv`, and the summary printed by `export.py` [EDIT: acquisition dates min-max, source resolution min/median/max, flagged counts, final class coverage from `results/uae_coverage.json`].
- Differences from MEET: different sensors and acquisition years, source resolution can differ from the nominal 1.0 m, no model retraining or tuning on UAE, small n per class, only classes with at least 5 accepted samples are evaluated.
- Attribution: Esri service copyright text saved in `uae/service_info.json` [EDIT: paste copyrightText]. Candidate tags: (c) OpenStreetMap contributors, ODbL.
- UAE imagery is private and never committed (`.gitignore`).

## Experiments
B1 custom CNN (center, end-to-end; architecture differs from B2, so B1 vs B2 is NOT a context comparison). B2 frozen MobileNetV2 + head (center). B3 concat fusion, B3-VD with view dropout. B4 gated fusion, B4-VD with view dropout (proposed). Heads use cached frozen MobileNetV2 (ImageNet) features. Comparisons: B2 vs B3, B3 vs B4, B3-VD vs B4-VD, B3 vs B3-VD, B4 vs B4-VD. All fusion variants are evaluated on all 7 non-empty view subsets, on MEET test and UAE.

## Reproducibility and platforms
- Seeds: 42 for data and B1; 42, 7, 123 for every head (mean +- std). Python, NumPy, and TF seeds are set in every script.
- Extraction and feature caching ran on [EDIT: Kaggle or Colab, GPU type]; everything else ran locally on CPU. Platform and TF version per split are in `features/features_meta.json`.
- Environment: `requirements.txt` (pinned), `requirements.lock.txt` (pip freeze). Run order: `run_all.sh`.
- Tuning: keras-tuner RandomSearch, 10 trials each for B3-VD and B4-VD, macro-F1 over the whole validation set; test and UAE never influence tuning.
- Statistics: exact McNemar on paired test predictions (per seed) with effect sizes; bootstrap 95% CI on per-class context gain, resampling test images within each class, paired across models.

## License caveats
- ImageNet-pretrained MobileNetV2 weights come from `keras.applications`; ImageNet terms apply.
- Esri imagery is subject to Esri terms of use; do not redistribute images.
- MEET terms apply to the subset; do not redistribute `subset/` unless the license permits.

## Results schema (`results/`)
- `runs.csv`: ds, variant, model, seed, subset, acc, top3, precision, recall, f1, roc_auc, pr_auc, ece. Subset is c, s, g, cs, cg, sg, csg for fusion heads and "all" for majority, chance, customcnn, b2. Variant is full24 (UAE also restricted).
- `table_{test|uae}_{full24|restricted}_{c|s|g|cs|cg|sg|csg}.csv`: model, `<metric>_mean`, `<metric>_std` for the 8 metrics, n_seeds. Baselines are included in every table.
- `per_class.csv` (also `per_class_arid.csv`, `per_class_other.csv`): class, precision, recall, f1, support, roc_auc, pr_auc, ds, variant, model, seed, group.
- `group_summary.csv`: ds, variant, model, group, precision, recall, f1.
- `errors.csv`: ds, model, seed, filename, true, pred, confidence (misclassified samples).
- `top_confusions.csv`: ds, model, true, pred, count, rate_of_true_class. `confusion_{ds}_{model}.csv`: matrix (seed 42).
- `mcnemar.csv`: a, b, seed, a_only_correct, b_only_correct, acc_a, acc_b, delta_acc, odds_ratio, p_value, significant, practical, verdict.
- `context_gain.csv`: model, seed, class, group, gain_f1, f1_lo, f1_hi, gain_acc, acc_lo, acc_hi. `context_gain_groups.csv`: model, seed, metric, gain_arid, gain_other, diff_arid_minus_other, diff_lo, diff_hi.
- `efficiency.csv`: model, trainable_params, total_params, backbone_params, train_seconds_mean, size_mb, latency_head_ms, latency_e2e_ms. `features_timing.json`: per split tf, device, platform, seconds, n, ms_per_image_per_view.
- `curves.json`: model -> grid, tpr, precision. `roc_pr_test.png`, `reliability_test.png`, `gate_dist.png`.
- `gates_samples.csv`, `gates_per_class.csv`, `gates_group.csv`, `gates_corr.csv`: per-sample gates, per-class and arid-vs-other means, gate-vs-correctness correlation.
- `gallery.json`, `gradcam_paths.json`, `gradcam/`: gallery metadata (3 correct + 2 wrong per class) and Grad-CAM overlays.
- `metrics.json`: seeds, subsets, arid, main_tables, group_summary, mcnemar, context_gain_groups, efficiency, features_timing, uae, notes.
- Training and data: `train_times.json`, `history.json`, `tuning.json`, `configs.json`, `head_val_summary.json`, `audit_a.json`, `audit_b.json`, `audit_dup_md5.csv`, `audit_b_top20.csv`, `uae_coverage.json`.

## Web harness
`python export.py b4vd 42`, then `python -m http.server 8000 -d web`. Toggle views to drive the mask and see the gates; the parity button checks the TF.js head against Keras.
```