import argparse, csv, hashlib, io, os, threading, time, zipfile
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from PIL import Image
from remotezip import RemoteZip
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from config import *
from common import make_views, vpath
ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=8); ap.add_argument("--classes",nargs="*",default=CLASSES); a=ap.parse_args()
a.classes=[c.replace("_"," ") for c in a.classes]; assert all(c in CLASSES for c in a.classes),a.classes
LOG=ROOT/"extract_log.csv"; new=not LOG.exists(); lock=threading.Lock(); tot=[0]; tok=os.environ.get("HF_TOKEN")
kw={"headers":{"Authorization":"Bearer "+tok}} if tok else {}
md5={r.filename:r.md5 for r in pd.read_csv(LOG).itertuples() if r.status=="ok"} if not new else {}
fh=open(LOG,"a",newline=""); w=csv.writer(fh)
if new: w.writerow(["class","filename","md5","status","info","attempts","bytes","secs"])
m=pd.read_csv(MAN); recs=m.to_dict("records")
def url(c): return f"https://huggingface.co/datasets/zinStar/MEET/resolve/main/{quote(c+'.zip')}"
def mmap(zf): return {os.path.basename(i.filename):i for i in zf.infolist() if i.filename.lower().endswith(".jpg")}  # match members by basename
def done(r): return r["filename"] in md5 and all(vpath(v,r["split"],r["class"],r["filename"]).exists() for v in VIEWS)
def emit(r,st,info="",att=1,n=0,t=0,h=""):
    with lock: w.writerow([r["class"],r["filename"],h,st,info,att,n,round(t,2)]); fh.flush(); tot[0]+=n; bar.update(1)
def proc(zf,mm,r):
    b=zf.read(mm[r["filename"]]); im=Image.open(io.BytesIO(b)).convert("RGB")
    for v,x in zip(VIEWS,make_views(im)): p=vpath(v,r["split"],r["class"],r["filename"]); p.parent.mkdir(parents=True,exist_ok=True); x.save(p,"JPEG",quality=95)
    return hashlib.md5(b).hexdigest(),len(b)
def work(cls,items):
    zf=mm=None; bad=[]
    for r in items:
        for k in range(3):
            try:
                if zf is None: zf=RemoteZip(url(cls),**kw); mm=mmap(zf)  # one RemoteZip per thread
                if r["filename"] not in mm: emit(r,"missing"); break
                t=time.time(); h,n=proc(zf,mm,r); md5[r["filename"]]=h; emit(r,"ok","remote",k+1,n,time.time()-t,h); break
            except Exception as e: zf=None; err=repr(e)[:100]; time.sleep(2**k)
        else: bad.append((r,err))
    if zf: zf.close()
    return bad
def fallback(cls,bad):
    p=hf_hub_download("zinStar/MEET",cls+".zip",repo_type="dataset",local_dir=ROOT/"tmp_zips")
    with zipfile.ZipFile(p) as zf:
        mm=mmap(zf)
        for r,_ in bad:
            try:
                if r["filename"] not in mm: emit(r,"missing"); continue
                t=time.time(); h,n=proc(zf,mm,r); md5[r["filename"]]=h; emit(r,"ok","fallback",4,n,time.time()-t,h)
            except Exception as e: emit(r,"fail",repr(e)[:100],4)
    os.remove(p)
def save(): m["md5"]=m["filename"].map(md5); m["ok"]=[done(r) for r in recs]; m.to_csv(MAN,index=False)
for cls in a.classes:
    todo=[r for r in recs if r["class"]==cls and not done(r)]; t0=time.time(); tot[0]=0; bar=tqdm(total=len(todo),desc=cls)
    with ThreadPoolExecutor(a.workers) as ex: bad=[x for l in ex.map(lambda c:work(cls,c),[todo[i::a.workers] for i in range(a.workers)]) for x in l]
    if bad and ZIP_GB[cls]<5:
        try: fallback(cls,bad)
        except Exception as e:
            for r,_ in bad: emit(r,"fail",repr(e)[:100],4)
    else:
        for r,e in bad: emit(r,"fail",e,3)
    bar.close(); save(); print(f"{cls}: {len(todo)} todo, {time.time()-t0:.0f}s, {tot[0]/1e6:.0f} MB read, {len(bad)} remote failures")
fh.close(); print("ok rows:",int(m["ok"].sum()),"/",len(m))