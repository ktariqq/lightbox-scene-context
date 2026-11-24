import sys, json, time, requests, numpy as np, pandas as pd, rasterio
from datetime import datetime, timezone
from PIL import Image
from rasterio.windows import Window
from rasterio.warp import transform
from tqdm import tqdm
from config import *
mode=sys.argv[1] if len(sys.argv)>1 else ""; assert mode in ("test","run"),"usage: 05_uae_fetch.py test|run"
S=requests.Session(); S.headers["User-Agent"]=UA
def scenes(lon,lat):  # least-cloudy Sentinel-2 L2A scenes over the point inside the fixed date window
    q={"collections":[S2_COLL],"intersects":{"type":"Point","coordinates":[lon,lat]},"datetime":S2_DATES,"query":{"eo:cloud_cover":{"lt":CLOUD_MAX}},"sortby":[{"field":"properties.eo:cloud_cover","direction":"asc"}],"limit":5}
    for k in range(3):
        try: r=S.post(S2_URL+"/search",json=q,timeout=60); r.raise_for_status(); time.sleep(DELAY); return r.json()["features"]
        except Exception as e: err=e; time.sleep(3*2**k)
    raise err
def window(href,lon,lat):  # NPX x NPX px (1280 m at 10 m) centered on the point; None if it leaves the tile or has no-data
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",AWS_NO_SIGN_REQUEST="YES"), rasterio.open(href) as s:
        x,y=transform("EPSG:4326",s.crs,[lon],[lat]); r,c=s.index(x[0],y[0]); h=NPX//2
        if min(r,c)-h<0 or r+h>s.height or c+h>s.width: return None
        a=s.read([1,2,3],window=Window(c-h,r-h,NPX,NPX))
    return None if (a.sum(0)==0).any() else a
def fetch(r):
    for f in scenes(r["lon"],r["lat"]):
        a=window(f["assets"][S2_ASSET]["href"],r["lon"],r["lat"])
        if a is not None: return f,a
    return None,None
def up(a): return Image.fromarray(a.transpose(1,2,0)).resize((1280,1280),Image.Resampling.BICUBIC)  # 128 px -> 1280 px so make_views is unchanged
def lap(a): g=a.mean(0).astype(np.float32); return float((4*g[1:-1,1:-1]-g[:-2,1:-1]-g[2:,1:-1]-g[1:-1,:-2]-g[1:-1,2:]).var())
UAE.mkdir(exist_ok=True); (UAE/"raw").mkdir(exist_ok=True); c=pd.read_csv(UAE/"candidates.csv")
if mode=="test":
    r0=c.iloc[0]; F=scenes(r0["lon"],r0["lat"]); assert F,"no scenes: check S2_COLL / S2_DATES in config.py"; print(len(F),"scenes, assets:",list(F[0]["assets"]))
    f,a=fetch(r0); assert a is not None,"no clean window for first candidate"; print(f["id"],f["properties"]["datetime"],"cloud",f["properties"].get("eo:cloud_cover"),a.shape,a.dtype,int(a.min()),int(a.max()))
    up(a).save(UAE/"raw"/f"{r0['id']}.jpg",quality=95); sys.exit()
MP=UAE/"meta.csv"; done=set(pd.read_csv(MP)["id"]) if MP.exists() else set()
for r in tqdm(c.to_dict("records")):
    if r["id"] in done: continue
    try:
        f,a=fetch(r)
        if a is None: open(UAE/"fetch_errors.txt","a").write(f"{r['id']} no clean scene\n"); continue
        up(a).save(UAE/"raw"/f"{r['id']}.jpg",quality=95); p=f["properties"]; cc=p.get("eo:cloud_cover",0.0)
        pd.DataFrame([{"id":r["id"],"acq_date":p["datetime"][:10],"src_name":f["id"],"src_res":float(S2_GSD),"blur_var":round(lap(a),2),"blur_flag":cc>=CLOUD_FLAG,"res_flag":S2_GSD>RES_MAX,"meta_json":json.dumps({"scene":f["id"],"collection":S2_COLL,"cloud_cover":cc})}]).to_csv(MP,mode="a",header=not MP.exists(),index=False)
    except Exception as e: open(UAE/"fetch_errors.txt","a").write(f"{r['id']} {e!r}\n")
m=pd.read_csv(MP); yrs=sorted({str(d)[:4] for d in m["acq_date"]})
json.dump({"copyrightText":"Contains modified Copernicus Sentinel data "+", ".join(yrs),"source":"Sentinel-2 L2A TCI via Element84 Earth Search (AWS Open Data)","collection":S2_COLL,"dates":S2_DATES,"fetched":datetime.now(timezone.utc).isoformat()},open(UAE/"service_info.json","w"),indent=1)
print(len(m),"fetched")