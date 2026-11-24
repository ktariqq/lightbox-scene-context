import json, time, requests, numpy as np, pandas as pd
from config import *
from common import cdir
F={"Mosque":'nwr["building"="mosque"]',"Swimming pool":'nwr["leisure"="swimming_pool"]',"Roundabout":'nwr["junction"="roundabout"]',"Airport":'nwr["aeroway"="aerodrome"]',"Port":'nwr["landuse"="port"]',"Resort":'nwr["tourism"="resort"]',"Beach":'nwr["natural"="beach"]',"Sand":'nwr["natural"="sand"]',"Rock":'nwr["natural"="bare_rock"]',"Scrub":'nwr["natural"="scrub"]',"Brownfield":'nwr["landuse"="brownfield"]',"Construction":'nwr["landuse"="construction"]',"Quarry":'nwr["landuse"="quarry"]',"Landfill":'nwr["landuse"="landfill"]',"Petroleum well":'nwr["man_made"="petroleum_well"]',"Solar power plant":'nwr["power"="plant"]["plant:source"="solar"]',"Thermal power plant":'nwr["power"="plant"]["plant:source"~"gas|coal|oil"]',"Chemical plant":'nwr["man_made"="works"]["product"~"chemical|petrochemical|fertili"]',"Substation":'nwr["power"="substation"]',"Highway":'way["highway"="motorway"]',"Mountain":'node["natural"="peak"]'}
F={k:v for k,v in F.items() if k in UAE_CLASSES}  # only classes resolvable at 10 m
URLS=["https://overpass-api.de/api/interpreter","https://overpass.kumi.systems/api/interpreter"]; UAE.mkdir(exist_ok=True)
def op(c):
    f=UAE/f"op_{cdir(c)}.json"
    if f.exists(): return json.load(open(f))
    q=f'[out:json][timeout:300];area["ISO3166-1"="AE"][admin_level=2]->.a;({F[c]}(area.a););out center tags;'
    for k in range(4):
        for u in URLS:
            try: r=requests.post(u,data={"data":q},headers={"User-Agent":UA},timeout=400); r.raise_for_status(); e=r.json()["elements"]; json.dump(e,open(f,"w")); time.sleep(5); return e
            except Exception as ex: print("retry",c,u,repr(ex)[:80]); time.sleep(10*(k+1))
    print("FAILED",c); return []
def hav(a,b,c,d): a,b,c,d=map(np.radians,(a,b,c,d)); h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2; return 12742*np.arcsin(np.sqrt(h))
pool={}
for c in F:
    for e in op(c):
        t=e.get("tags",{}); p=e if "lat" in e else e.get("center")
        if p is None or "military" in t or t.get("landuse")=="military": continue  # OSM is candidate generation only
        pool.setdefault(c,[]).append((f'{cdir(c)}_{e["type"]}{e["id"]}',c,p["lat"],p["lon"],json.dumps(t,ensure_ascii=False)))
rng=np.random.default_rng(SEED); LA,LO,rows=[],[],[]
for c in sorted(pool,key=lambda c:len(pool[c])):  # rare classes pick first; global 2.5 km separation
    n=0
    for i in rng.permutation(len(pool[c])):
        r=pool[c][i]
        if LA and hav(r[2],r[3],np.array(LA),np.array(LO)).min()<2.5: continue
        LA.append(r[2]); LO.append(r[3]); rows.append(r); n+=1
        if n==N_UAE: break
    print(f"{c}: pool {len(pool[c])}, kept {n}")
print("no candidates:",[c for c in F if c not in pool],"| not sampled:",[c for c in CLASSES if c not in F])
pd.DataFrame(rows,columns=["id","class","lat","lon","tags"]).sort_values(["class","id"]).to_csv(UAE/"candidates.csv",index=False)