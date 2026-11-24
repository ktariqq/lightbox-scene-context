import json, pandas as pd
from huggingface_hub import hf_hub_download
from config import *
from common import parse_line
DATA.mkdir(exist_ok=True); parts,av=[],{}
for sp,fn in [("train","MEET_train_1125.txt"),("val","MEET_valid_1125.txt"),("test","MEET_test_1125.txt")]:
    p=hf_hub_download("zinStar/MEET",fn,repo_type="dataset",local_dir=DATA)
    d=pd.DataFrame([parse_line(l) for l in open(p,encoding="utf8") if l.strip()],columns=["filename","class","idx"]); d=d[d["class"].isin(CLASSES)]
    av[sp]=d.groupby("class").size()  # official split lists only; sample per class with fixed seed
    parts.append(pd.concat([g.sample(min(SPLITS[sp],len(g)),random_state=SEED) for _,g in d.groupby("class")]).assign(split=sp))
m=pd.concat(parts)[["filename","class","split"]].sort_values(["split","class","filename"]).reset_index(drop=True)
assert m["class"].nunique()==len(CLASSES),set(CLASSES)-set(m["class"])
m.to_csv(MAN,index=False)
cnt=m.groupby(["class","split"]).size().unstack(fill_value=0).reindex(CLASSES,fill_value=0)
out=cnt.join(pd.DataFrame(av).reindex(CLASSES).add_suffix("_avail")); out.to_csv(ROOT/"subset_counts.csv"); print(out)
for sp,n in SPLITS.items():
    if (cnt[sp]!=n).any(): print(f"IMBALANCE {sp}: target {n}, classes below target:",cnt.index[cnt[sp]!=n].tolist())
    if cnt[sp].nunique()>1: print(f"UNEVEN {sp}: class weights needed")
print(f"est download ~{len(m)*EST_MB/1024:.1f} GB ({EST_MB} MB/image assumed)"); print((cnt.sum(axis=1)*EST_MB/1024).round(2).to_string())
cj=ROOT/"classes.json"
if not cj.exists(): json.dump({"classes":CLASSES,"arid":[]},open(cj,"w"),indent=1)
c=json.load(open(cj)); assert c["classes"]==CLASSES; print("EDIT: fill the arid list in classes.json before seeing results" if not c["arid"] else f"arid list: {c['arid']}")