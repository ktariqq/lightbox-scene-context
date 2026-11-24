# AIRE410 · Zero-G Lightbox

Static site: a scroll story, an Evidence section and an in-browser instrument (TF.js 4.22.0). No build step.

## Layout
- `index.html`, `css/`, `js/`, `assets/`: the site
- `model/`: a verbatim copy of the export's `web/` folder (`manifest.json`, `data.json`, `gradcam_paths.json`, `parity.json`, model files, gallery and Grad-CAM images)
- `data/results/`: a verbatim copy of `results/` (CSV, JSON and PNG files)

## Regenerate model and data
```bash
python export.py b4vd 42          # writes web/
cp -r web/*     model/
cp -r results/* data/results/
```
Do not copy anything from `uae/` or any Esri imagery. The site shows UAE only as aggregate numbers.

## Deploy on GitHub Pages
1. Push the repo with `.nojekyll` at the root.
2. Settings → Pages → Deploy from branch → `main` / root.
3. Open `https://<user>.github.io/<repo>/`. All paths are relative, so the `/<repo>/` base path works.

## Local preview
```bash
python -m http.server 8000      # then open http://localhost:8000/
```
Opening `index.html` from disk will not work, because the browser blocks fetch and model loading on `file://`.

## Optional export for live Grad-CAM
If the Keras backbone is a flat functional model with a layer named `out_relu`, the site derives the split model itself. Otherwise add this to `manifest.json` and export a backbone with outputs `[conv 7×7×1280, pooled 1280]`:
```json
"backbone_cam": { "format": "graph", "path": "backbone_cam/model.json", "inputs": ["<input name>"] }
```

## Data honesty
Every number comes from `data/results/` or `model/`. Missing files show a “results not exported yet” state. Placeholders from the README appear as TODO chips in the Method section.