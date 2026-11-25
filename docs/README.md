# Lightbox web app

Static site with no build step: a hero with a three-plate view stack, an in-browser model, and a four-chapter study. TensorFlow.js 4.22.0 loads from jsDelivr.

## Layout
- `index.html`, `404.html`, `css/`, `js/`, `assets/`: the site
- `manifest.json`, `data.json`, `gradcam_paths.json`, `parity.json`: written by `export.py`
- `head.json`, `head.bin`: gated head weights, written by `export_web.py`
- `tfjs/backbone/`: MobileNetV2 as a TF.js graph model
- `img/`, `cam/`: gallery samples and Grad-CAM overlays, copied by `export.py`
- `data/results/`: copied from `results/` by the deploy workflow and ignored by git

## Regenerate the model files
Run from the repository root, after `evaluate.py` and `explain.py`:
```bash
./run_all.sh web
```
This runs `export_web.py b4vd 42` (backbone SavedModel, head weights, numpy parity check), converts the backbone with `tensorflowjs_converter`, then runs `export.py b4vd 42`. Install the converter in its own environment: `pip install tensorflowjs==4.22.0 -c tfjs-constraints.txt`.

## Local preview
```bash
mkdir -p docs/data && cp -r results docs/data/results
python -m http.server 8000 -d docs     # open http://localhost:8000/
```
Opening `index.html` from disk does not work: browsers block fetch and model loading on `file://`.

## Deploy
Settings → Pages → Source: **GitHub Actions**. Every push to `main` runs `.github/workflows/deploy.yml`, which copies `results/` into `docs/data/results/` and publishes `docs/`. The deploy does not block on missing files; a missing file shows a "results not exported yet" state on the site instead. All paths are relative, so the `/<repo>/` base path works.

## What runs where
- The head (gated fusion) is parity-tested against Keras: `export_web.py` asserts agreement within 1e-3 and the page has a "run the check" button.
- Images are resized with the browser canvas, which is not bit-identical to the Python bicubic resize. Only the head is parity-tested, so live top-1 can differ slightly from the stored prediction.
- Live Grad-CAM on uploads needs a backbone export that returns the 7x7x1280 conv map. The graph export does not, so gallery samples use precomputed overlays and uploads show a notice.

## Data honesty
Every number on the site is read from `data/results/` or `data.json`. A missing file shows a "results not exported yet" state. Only aggregate UAE numbers appear; no UAE imagery is published.