import sys, io, re, json, math, time, requests, numpy as np, pandas as pd
from datetime import datetime, timezone
from PIL import Image
from tqdm import tqdm
from config import *
B="https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer"; S=requests.Session(); S.headers["User-Agent"]=UA
mode=sys.argv[1] if len(sys.argv)>1 else ""; assert mode in ("test","run"),"usage: 05_uae_fetch.py test|run"
def get(u,**p):
    for k in range(3):
        try: r=S.get(u,params=p,timeout=90); r.raise_for_status(); time.sleep(DELAY); return r
        except Exception as e: err=e; time.sleep(3*2**k)
    raise err
def bbox(la,lo): x=6378137*math.radians(lo); y=6378137*math.log(math.tan(math.pi/4+math.radians(la)/2)); h=640/math.cos(math.radians(la)); return x,y,f"{x-h:.3f},{y-h:.3f},{x+h:.3f},{y+h:.3f}"  # 1280 m ground footprint
def img(la,lo):
    r=get(B+"/export",bbox=bbox(la,lo)[2],bboxSR=3857,imageSR=3857,size="1280,1280",format="jpg",f="image")
    assert r.headers.get("content-type","").startswith("image"),r.text[:300]
    im=Image.open(io.BytesIO(r.content)); assert im.size==(1280,1280),im.size; return r,im
def fdate(v): return datetime.fromtimestamp(v/1000,timezone.utc).date().isoformat() if isinstance(v,(int,float)) and v>1e10 else str(v)
def meta(x,y): return {l:get(f"{B}/{l}/query",geometry=f"{x},{y}",geometryType="esriGeometryPoint",inSR=3857,spatialRel="esriSpatialRelIntersects",outFields="*",returnGeometry="false",f="json").json() for l in layers}
def parse(out):
    d={}
    for j in out.values():
        for f in (j.get("features",[]) if isinstance(j,dict) else []):
            for k,v in f.get("attributes",{}).items():
                k=k.lower(); num=re.search(r"\d+\.?\d*",str(v))
                if v in (None,"",0,-1): continue
                if "date" in k: d.setdefault("acq_date",fdate(v))
                elif "res" in k and num: d.setdefault("src_res",float(num.group()))
                elif "name" in k or "desc" in k or "source" in k: d.setdefault("src_name",str(v))
    return d
def lap(im): g=np.asarray(im.convert("L"),dtype=np.float32); return float((4*g[1:-1,1:-1]-g[:-2,1:-1]-g[2:,1:-1]-g[1:-1,:-2]-g[1:-1,2:]).var())
UAE.mkdir(exist_ok=True); (UAE/"raw").mkdir(exist_ok=True); c=pd.read_csv(UAE/"candidates.csv")
info=get(B,f="json").json(); layers=[l["id"] for l in info.get("layers",[])]
json.dump({"copyrightText":info.get("copyrightText"),"layers":info.get("layers"),"fetched":datetime.now(timezone.utc).isoformat(),"url":B},open(UAE/"service_info.json","w"),indent=1)
if mode=="test":
    r0=c.iloc[0]; r,im=img(r0["lat"],r0["lon"]); print(r.status_code,r.headers.get("content-type"),len(r.content),"bytes",im.size,r.url)
    x,y,_=bbox(r0["lat"],r0["lon"]); o=meta(x,y); print("copyright:",info.get("copyrightText")); print("layers:",info.get("layers")); print(json.dumps(o,indent=1)[:3000]); print("parsed:",parse(o))
    (UAE/"raw"/f"{r0['id']}.jpg").write_bytes(r.content); sys.exit()
MP=UAE/"meta.csv"; done=set(pd.read_csv(MP)["id"]) if MP.exists() else set(); first=True
for r in tqdm(c.to_dict("records")):
    if r["id"] in done: continue
    try:
        p=UAE/"raw"/f"{r['id']}.jpg"
        if p.exists(): im=Image.open(p)
        else: rr,im=img(r["lat"],r["lon"]); p.write_bytes(rr.content)
        x,y,_=bbox(r["lat"],r["lon"]); o=meta(x,y)
        if first: print(json.dumps(o,indent=1)[:3000]); first=False
        d=parse(o); v=lap(im); sr=d.get("src_res")
        pd.DataFrame([{"id":r["id"],"acq_date":d.get("acq_date"),"src_name":d.get("src_name"),"src_res":sr,"blur_var":round(v,2),"blur_flag":v<BLUR_MIN,"res_flag":bool(sr and sr>RES_MAX),"meta_json":json.dumps(o)}]).to_csv(MP,mode="a",header=not MP.exists(),index=False)
    except Exception as e: open(UAE/"fetch_errors.txt","a").write(f"{r['id']} {e!r}\n")