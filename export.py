import sys
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from pathlib import Path

from config import *
from common import cdir, seed_all

tf.config.set_visible_devices([], "GPU")
seed_all()

name = sys.argv[1] if len(sys.argv) > 1 else "b4vd"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
assert name in ("b3", "b3vd", "b4", "b4vd"), name

W = ROOT / "docs"
T = W / "tfjs"
X = ROOT / "tmp_export"
T.mkdir(parents=True, exist_ok=True)
X.mkdir(exist_ok=True)

# Convert a Keras model to TF.js.
def conv(m, tag):
    o = T / tag
    f = X / f"{tag}.keras"
    m.save(f)
    r = subprocess.run(
        ["tensorflowjs_converter", "--input_format=keras_keras", str(f), str(o)],
        capture_output=True,
        text=True
    )  # try the .keras file first
    if r.returncode == 0 and (o / "model.json").exists():
        return {"path": f"tfjs/{tag}/model.json", "format": "layers", "inputs": [t.name for t in m.inputs]}

    print(tag, "keras_keras failed, falling back to SavedModel:", r.stderr[-300:])
    shutil.rmtree(o, ignore_errors=True)
    sd = X / f"{tag}_sm"
    shutil.rmtree(sd, ignore_errors=True)
    m.export(str(sd))
    r = subprocess.run(
        ["tensorflowjs_converter", "--input_format=tf_saved_model", "--output_format=tfjs_graph_model", "--signature_name=serving_default", "--saved_model_tags=serve", str(sd), str(o)],
        capture_output=True,
        text=True
    )
    assert r.returncode == 0, r.stderr[-500:]
    return {
        "path": f"tfjs/{tag}/model.json",
        "format": "graph",
        "inputs": list(tf.saved_model.load(str(sd)).signatures["serving_default"].structured_input_signature[1].keys())
    }

# Prepare backbone and head models.
bb = keras.models.load_model(MODELS / "mobilenetv2_backbone.keras", compile=False)
i = keras.Input((224, 224, 3), name="image")
bbx = keras.Model(
    i,
    bb(keras.layers.Rescaling(1 / 127.5, offset=-1)(i)),
    name="backbone"
)  # in-model preprocess_input: [0,255] -> [-1,1], one shared backbone run once per view

full = keras.models.load_model(MODELS / f"{name}_s{seed}.keras", compile=False)
o = full.outputs
hd = keras.Model(
    full.inputs,
    [o[0], o[2]] if len(o) == 3 else o[0],
    name="head"
)  # head takes (3 feature vectors, mask), returns (logits, per-view gates)

# Backbone TF.js conversion is skipped: it was already produced (docs/tfjs/backbone/model.json, Step 1)
# and conv() needs the tensorflowjs CLI on PATH. Head is provided by export_web (head.bin).
man = {
    "model": name,
    "seed": seed,
    "views": VIEWS,
    "backbone": {"path": "tfjs/backbone/model.json", "format": "graph", "inputs": ["image"]},
    "head": {"has_gates": len(o) == 3},
}
json.dump(man, open(W / "manifest.json", "w"), indent=1)

# Parity check data for one cached test sample.
Xf = {v: np.load(FEAT / f"{v}_test.npy")[:1] for v in VIEWS}
cases = []
for mk in ([1, 1, 1], [1, 0, 1], [0, 1, 0]):
    r = hd.predict({**Xf, "mask": np.array([mk], "float32")}, verbose=0)
    r = r if isinstance(r, list) else [r]
    cases.append({
        "mask": mk,
        "logits": r[0][0].tolist(),
        "gates": r[1][0].tolist() if len(r) > 1 else None
    })

json.dump(
    {**{v: Xf[v][0].tolist() for v in VIEWS}, "cases": cases},
    open(W / "parity.json", "w")
)

rd = lambda n: pd.read_csv(RES / n)
rec = lambda d: json.loads(d.to_json(orient="records"))
cj = json.load(open(ROOT / "classes.json"))
ar = set(cj["arid"])

# Per-class metrics for the UI.
pc = rd("per_class.csv")
pc = pc[(pc["ds"] == "test") & (pc["variant"] == "full24")]
PCL = {
    m: rec(pc[pc["model"] == m].groupby("class")[["precision", "recall", "f1", "support", "roc_auc", "pr_auc"]].mean().reindex(CLASSES).reset_index())
    for m in ("customcnn", "b2", name)
}

# Context gain for this seed.
cg = rd("context_gain.csv")
cg = rec(cg[(cg["model"] == name) & (cg["seed"] == seed)])

cm = pd.read_csv(RES / f"confusion_test_{name}.csv", index_col=0)
G = json.load(open(RES / "gallery.json"))

# Copy gallery images and Grad-CAM overlays into docs/.
for it in G["items"]:
    stem = it["filename"][:-4]
    for v in VIEWS:
        a = it["views"][v]
        di = Path(f"img/{v}/{cdir(it['class'])}/{stem}.jpg")
        dc = Path(f"cam/{cdir(it['class'])}/{stem}_{v}.png")
        for s, d in ((a["image"], di), (a["cam"], dc)):
            (W / d).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / s, W / d)
        a["image"], a["cam"] = di.as_posix(), dc.as_posix()

json.dump(
    {it["filename"]: {v: it["views"][v]["cam"] for v in VIEWS} for it in G["items"]},
    open(W / "gradcam_paths.json", "w"),
    indent=1
)

mt = json.load(open(RES / "metrics.json"))["main_tables"]
u = pd.read_csv(UMAN) if UMAN.exists() else None
US = None if u is None else {
    "n": len(u),
    "acq_date_min": str(u["acq_date"].dropna().min()),
    "acq_date_max": str(u["acq_date"].dropna().max()),
    "src_res_min": float(u["src_res"].min()),
    "src_res_median": float(u["src_res"].median()),
    "src_res_max": float(u["src_res"].max()),
    "flag_blur": int(u["blur_flag"].sum()),
    "flag_coarse": int(u["res_flag"].sum()),
    "per_class": u["class"].value_counts().to_dict(),
    "copyright": json.load(open(UAE / "service_info.json"))["copyrightText"]
}

D = {
    "classes": CLASSES,
    "groups": {
        "arid": [c for c in CLASSES if c in ar],
        "other": [c for c in CLASSES if c not in ar]
    },
    "head_model": name,
    "seed": seed,
    "per_class": PCL,
    "confusion": {"seed": 42, "labels": CLASSES, "matrix": cm.values.tolist()},
    "context_gain": cg,
    "efficiency": rec(rd("efficiency.csv")),
    "summary_test": mt.get("test_full24"),
    "summary_uae": mt.get("uae_full24"),
    "uae_summary": US,
    "gallery": G
}

open(W / "data.json", "w").write(
    json.dumps(D, default=lambda x: x.item() if hasattr(x, "item") else str(x)).replace("NaN", "null")
)
shutil.rmtree(X, ignore_errors=True)
print("UAE summary for README:", json.dumps(US, indent=1))
print(
    {p.name: round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 2) for p in (T / "backbone", T / "head")},
    "MB"
)