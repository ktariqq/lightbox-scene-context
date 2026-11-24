import sys, json, numpy as np, pandas as pd
from config import *
from common import vpath
st=sys.argv[1] if len(sys.argv)>1 else "a"; RES.mkdir(exist_ok=True); m=pd.read_csv(MAN); out={}
if st=="a":
    out["rows"],out["ok_rows"]=len(m),int(m["ok"].sum())
    for v in VIEWS:  # every sample has all 3 views, stored under its own split, no strays
        want={str(vpath(v,s,c,f)) for f,c,s in zip(m["filename"],m["class"],m["split"])}; have={str(p) for p in (SUBSET/v).rglob("*.jpg")}
        out[f"{v}_missing"],out[f"{v}_extra"]=len(want-have),len(have-want)
    k=m.dropna(subset=["md5"]); g=k.groupby("md5")["split"].nunique(); d=k[k["md5"].isin(g[g>1].index)].sort_values("md5")
    d.to_csv(RES/"audit_dup_md5.csv",index=False); out["dup_md5_groups_across_splits"]=int((g>1).sum()); out["dup_md5_repeated_rows"]=int(k["md5"].duplicated().sum())
    out["class_counts"]=m.groupby(["class","split"]).size().unstack(fill_value=0).to_dict("index")
    out["geographic_overlap"]="NOT CHECKABLE: MEET has no coordinates or location metadata; near-duplicate scenes are screened by feature similarity (stage b)"
    json.dump(out,open(RES/"audit_a.json","w"),indent=1,default=int)
else:
    QS=[0,.5,.9,.95,.99,1]; rows=[]
    for v in ("center","global"):
        T=np.load(FEAT/f"{v}_train.npy"); T/=np.linalg.norm(T,axis=1,keepdims=True); ti=pd.read_csv(FEAT/"train_index.csv")
        for sp in ("val","test"):
            Q=np.load(FEAT/f"{v}_{sp}.npy"); Q/=np.linalg.norm(Q,axis=1,keepdims=True); qi=pd.read_csv(FEAT/f"{sp}_index.csv")
            S=Q@T.T; j=S.argmax(1); s=S.max(1)  # cosine to nearest TRAIN sample
            out[f"{v}_{sp}"]={"n":len(s),"quantiles":dict(zip(map(str,QS),np.quantile(s,QS).tolist())),"n_above_0.95":int((s>.95).sum())}
            rows+=[(v,sp,str(vpath(v,sp,CLASSES[qi.label[i]],qi.filename[i])),str(vpath(v,"train",CLASSES[ti.label[j[i]]],ti.filename[j[i]])),float(s[i])) for i in range(len(s))]
    pd.DataFrame(rows,columns=["view","split","query","nearest_train","cos"]).sort_values("cos",ascending=False).groupby("view").head(20).to_csv(RES/"audit_b_top20.csv",index=False)
    json.dump(out,open(RES/"audit_b.json","w"),indent=1)
print(json.dumps(out,indent=1,default=int))