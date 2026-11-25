import sys, json, shutil, numpy as np, pandas as pd, keras
from PIL import Image
from config import *
from common import vpath
name=sys.argv[1] if len(sys.argv)>1 else "b4vd"; seed=int(sys.argv[2]) if len(sys.argv)>2 else 42; assert name in ("b4","b4vd")
W,X=ROOT/"web",ROOT/"tmp_export"; W.mkdir(exist_ok=True); shutil.rmtree(X,ignore_errors=True); X.mkdir()
# backbone: same [0,255] -> [-1,1] preprocessing as features.py
i=keras.Input((224,224,3),name="image"); bb=keras.applications.MobileNetV2((224,224,3),include_top=False,weights="imagenet",pooling="avg")
bbx=keras.Model(i,bb(keras.layers.Rescaling(1/127.5,offset=-1)(i))); bbx.export(str(X/"backbone_sm"))
ix=pd.read_csv(FEAT/"test_index.csv"); F={v:np.load(FEAT/f"{v}_test.npy") for v in VIEWS}; p=lambda k:vpath("center","test",CLASSES[ix.label[k]],ix.filename[k])
a=bbx.predict(np.stack([np.asarray(Image.open(p(k)).convert("RGB"),"float32") for k in range(8)]),verbose=0); b=F["center"][:8]
assert ((a*b).sum(1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1)>.999).all(),"backbone differs from cached features"
# head: dump weights, then check a numpy forward pass against Keras
m=keras.models.load_model(MODELS/f"{name}_s{seed}.keras",compile=False); Ls={l.name:l for l in m.layers}
nm=[f"{a}_{v}" for a in ("proj","gate") for v in VIEWS]+["logits"]
h=[l for l in m.layers if isinstance(l,keras.layers.Dense) and l.name not in nm and l.kernel.shape[0]>3]; assert len(h)==1; Ls["hid"]=h[0]
A={f"{n}/{t}":getattr(Ls[n],a).numpy().astype("<f4") for n in nm[:-1]+["hid","logits"] for t,a in (("k","kernel"),("b","bias"))}
o,M=0,{}
for k,v in A.items(): M[k]={"shape":list(v.shape),"offset":o}; o+=v.size
np.concatenate([v.ravel() for v in A.values()]).tofile(W/"head.bin"); json.dump({"views":VIEWS,"classes":CLASSES,"w":M},open(W/"head.json","w"))
r=lambda x:np.maximum(x,0); d=lambda x,n:x@A[n+"/k"]+A[n+"/b"]
def fwd(F,mk):
    pm=[r(d(F[v],"proj_"+v))*mk[i] for i,v in enumerate(VIEWS)]; fl=np.concatenate(pm,1)
    g=[1/(1+np.exp(-d(fl,"gate_"+v)))*mk[i] for i,v in enumerate(VIEWS)]
    return d(r(d(sum(x*y for x,y in zip(g,pm))/sum(mk),"hid")),"logits"),np.stack([x.mean(1) for x in g],1)
S={v:F[v][:256] for v in VIEWS}; cases=[]
for mk in ([1,1,1],[1,0,1],[0,1,0],[1,1,0]):
    k=m.predict({**S,"mask":np.tile(np.float32(mk),(256,1))},verbose=0); lg,gt=fwd(S,mk)
    assert np.allclose(k[0],lg,atol=1e-3) and np.allclose(k[2],gt,atol=1e-3),f"head mismatch {mk}"
    cases.append({"mask":mk,"logits":lg[0].tolist(),"gates":gt[0].tolist()})
json.dump({**{v:F[v][0].tolist() for v in VIEWS},"cases":cases},open(W/"parity.json","w")); shutil.copy(p(0),W/"parity_center.jpg"); print("OK: now convert tmp_export/backbone_sm")