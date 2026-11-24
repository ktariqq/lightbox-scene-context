import sys
import os
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from config import *
from common import vpath, cdir, seed_all

# Seed everything and create the feature cache directory.
seed_all()
FEAT.mkdir(exist_ok=True)

# Splits to process; defaults to train/val/test.
sps = sys.argv[1:] or ["train", "val", "test"]

# Detect platform and GPU availability for metadata.
plat = os.environ.get("AIRE_PLATFORM") or ("kaggle" if os.path.exists("/kaggle") else "local")
gpu = bool(tf.config.list_physical_devices("GPU"))

# Frozen MobileNetV2 backbone: GAP to a 1280-d feature vector.
bb = tf.keras.applications.MobileNetV2((224, 224, 3), include_top=False, weights="imagenet", pooling="avg")
pre = tf.keras.applications.mobilenet_v2.preprocess_input

# Build a batched dataset from image paths.
def mk(P):
    return (
        tf.data.Dataset.from_tensor_slices(P)
        .map(
            lambda p: tf.ensure_shape(
                pre(tf.cast(tf.io.decode_jpeg(tf.io.read_file(p), channels=3), tf.float32)),
                [224, 224, 3]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(64)
        .prefetch(1)
    )

# Load or initialize feature metadata.
mp = FEAT / "features_meta.json"
meta = json.load(open(mp)) if mp.exists() else {}

for sp in sps:
    # Resolve sample paths and labels for this split.
    if sp == "uae":
        u = pd.read_csv(UMAN)
        idx = pd.DataFrame({"filename": u["id"], "label": [CLASSES.index(c) for c in u["class"]]})
        P = {v: [str(UAE_SUBSET / v / cdir(c) / f"{i}.jpg") for i, c in zip(u["id"], u["class"])] for v in VIEWS}
    elif sp == "meet10":  # same samples and order as the test split, degraded views
        d = pd.read_csv(MAN)
        d = d[d["split"] == "test"]
        idx = pd.DataFrame({"filename": d["filename"].values, "label": [CLASSES.index(c) for c in d["class"]]})
        P = {v: [str(MEET10 / v / cdir(c) / (Path(f).stem + ".jpg")) for f, c in zip(d["filename"], d["class"])] for v in VIEWS}
    else:
        d = pd.read_csv(MAN)
        d = d[d["split"] == sp]
        assert d["ok"].all(), f"{sp}: samples not extracted"  # never silently drop samples
        idx = pd.DataFrame({"filename": d["filename"].values, "label": [CLASSES.index(c) for c in d["class"]]})
        P = {v: [str(vpath(v, sp, c, f)) for f, c in zip(d["filename"], d["class"])] for v in VIEWS}

    # All three views must stay aligned per sample.
    assert all([Path(p).stem for p in P[v]] == [Path(p).stem for p in P["center"]] for v in VIEWS)

    # Extract and save features for each view.
    t = time.time()
    for v in VIEWS:
        F = bb.predict(mk(P[v]), verbose=0)
        assert F.shape == (len(idx), 1280) and np.isfinite(F).all()
        np.save(FEAT / f"{v}_{sp}.npy", F.astype("float32"))

    # Save sample index and timing metadata.
    idx.to_csv(FEAT / f"{sp}_index.csv", index=False)
    meta[sp] = {
        "tf": tf.__version__,
        "device": "gpu" if gpu else "cpu",
        "platform": plat,
        "seconds": round(time.time() - t, 1),
        "n": len(idx)
    }
    json.dump(meta, open(mp, "w"), indent=1)
    print(sp, meta[sp])