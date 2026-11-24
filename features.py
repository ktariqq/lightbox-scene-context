import sys, os, json, time, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from config import *
from common import vpath, cdir, seed_all
seed_all(); FEAT.mkdir(exist_ok=True); sps=sys.argv[1:] or ["train","val","test"]
plat=os.environ.get("AIRE_PLATFORM") or ("kaggle" if os.path.exists("/kaggle") else "local"); gpu=bool(tf.config.list_physical_devices("GPU"))
bb=tf.keras.applications.MobileNetV2((224,224,3),include_top=False,weights="imagenet",pooling="avg")  # frozen backbone, GAP -> 1280-d
pre=tf.keras.applications.mobilenet_v2.preprocess_input
def mk(P): return tf.data.Dataset.from_tensor_slices(P).map(lambda p:tf.ensure_shape(pre(tf.cast(tf.io.decode_jpeg(tf.io.read_file(p),channels=3),tf.float32)),[224,224,3]),num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(1)
mp=FEAT/"features_meta.json"; meta=json.load(open(mp)) if mp.exists() else {}
for sp in sps:
    if sp=="uae":
        u=pd.read_csv(UMAN); idx=pd.DataFrame({"filename":u["id"],"label":[CLASSES.index(c) for c in u["class"]]})
        P={v:[str(UAE_SUBSET/v/cdir(c)/f"{i}.jpg") for i,c in zip(u["id"],u["class"])] for v in VIEWS}
    else:
        d=pd.read_csv(MAN); d=d[d["split"]==sp]; assert d["ok"].all(),f"{sp}: samples not extracted"  # never silently drop samples
        idx=pd.DataFrame({"filename":d["filename"].values,"label":[CLASSES.index(c) for c in d["class"]]}); P={v:[str(vpath(v,sp,c,f)) for f,c in zip(d["filename"],d["class"])] for v in VIEWS}
    assert all([Path(p).stem for p in P[v]]==[Path(p).stem for p in P["center"]] for v in VIEWS)  # 3 views aligned
    t=time.time()
    for v in VIEWS:
        F=bb.predict(mk(P[v]),verbose=0); assert F.shape==(len(idx),1280) and np.isfinite(F).all(); np.save(FEAT/f"{v}_{sp}.npy",F.astype("float32"))
    idx.to_csv(FEAT/f"{sp}_index.csv",index=False); meta[sp]={"tf":tf.__version__,"device":"gpu" if gpu else "cpu","platform":plat,"seconds":round(time.time()-t,1),"n":len(idx)}
    json.dump(meta,open(mp,"w"),indent=1); print(sp,meta[sp])