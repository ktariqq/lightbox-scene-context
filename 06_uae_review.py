import argparse, json, pandas as pd
from PIL import Image
from config import *
ap=argparse.ArgumentParser(); ap.add_argument("--only"); ap.add_argument("--out",default="review.html"); ap.add_argument("--blind",action="store_true"); a=ap.parse_args()
c=pd.read_csv(UAE/"candidates.csv").merge(pd.read_csv(UAE/"meta.csv").drop(columns=["meta_json","src_name","blur_var"]),on="id")
if a.only: c=c[c["id"].isin(pd.read_csv(a.only)["id"])]
if not a.blind: c=c.sort_values(["class","id"])
(UAE/"thumbs").mkdir(exist_ok=True)
for i in c["id"]:
    t=UAE/"thumbs"/f"{i}.jpg"
    if not t.exists(): Image.open(UAE/"raw"/f"{i}.jpg").convert("RGB").resize((400,400)).save(t,quality=85)
H=r'''<!doctype html><meta charset=utf-8><title>UAE review</title><style>
body{margin:0;color:#e6e1f5;font:13px system-ui;background:#0b1026 radial-gradient(1px 1px at 20px 30px,#fff8,transparent),radial-gradient(1px 1px at 90px 140px,#fff6,transparent),radial-gradient(1.5px 1.5px at 160px 70px,#fffa,transparent);background-size:200px 200px}
header{position:sticky;top:0;z-index:9;display:flex;gap:12px;align-items:center;padding:8px 16px;background:#0b1026ee;border-bottom:1px solid #663399}
button{background:#663399;color:#fff;border:0;padding:6px 12px;border-radius:4px;cursor:pointer}h2{color:#b794f6;margin:16px}
.g{display:flex;flex-wrap:wrap;gap:10px;padding:0 16px}.c{width:220px;border:1px solid #2a3566;padding:6px;border-radius:4px}.t{position:relative}.t img{width:100%;display:block}
.b1,.b2{position:absolute;border:1px solid #663399;box-sizing:border-box}.b1{left:40%;top:40%;width:20%;height:20%}.b2{left:20%;top:20%;width:60%;height:60%;border-style:dashed}
select{width:100%;background:#0b1026;color:#e6e1f5;border:1px solid #663399;margin-top:4px}.f{color:#f2c14e}</style>
<header><svg width=34 height=34><g transform="rotate(-25 17 17)"><ellipse cx=17 cy=17 rx=15 ry=6 fill=none stroke=#663399 /><rect x=30 y=15 width=4 height=4 fill=#b794f6 /></g><circle cx=17 cy=17 r=4 fill=#663399 /></svg><b>UAE review</b><span id=n></span><button onclick=dl()>Download labels.csv</button></header><div id=r></div><script>
const D=__D__,C=__C__,BL=__B__,K="lab"+location.pathname,S=JSON.parse(localStorage[K]||"{}"),R=document.getElementById("r");let cur="",g;
D.forEach(d=>{if(!g||(!BL&&d.class!=cur)){cur=d.class;R.insertAdjacentHTML("beforeend",`<h2>${BL?"":cur}</h2><div class=g></div>`);g=R.lastChild}
g.insertAdjacentHTML("beforeend",`<div class=c><a href="raw/${d.id}.jpg" target=_blank><div class=t><img loading=lazy src="thumbs/${d.id}.jpg"><div class=b1></div><div class=b2></div></div></a><div>${d.id}</div><div class="${d.blur_flag||d.res_flag?"f":""}">${d.acq_date||"-"} | ${d.src_res??"-"} m${d.blur_flag?" | blur":""}${d.res_flag?" | coarse":""}</div><select data-id="${d.id}"><option value="">pending<option value=reject>reject${BL?"":`<option value=accept>accept (${d.class})`}${C.map(x=>`<option>${x}`).join("")}</select></div>`)});
const cnt=()=>document.getElementById("n").textContent=Object.values(S).filter(Boolean).length+"/"+D.length+" reviewed";
document.querySelectorAll("select").forEach(s=>{s.value=S[s.dataset.id]||"";s.onchange=()=>{S[s.dataset.id]=s.value;localStorage[K]=JSON.stringify(S);cnt()}});cnt();
function dl(){const L=["id,osm_class,decision,final_class"];D.forEach(d=>{const v=S[d.id]||"",f=v=="accept"?d.class:v,dc=!v?"":v=="reject"?"reject":f==d.class?"accept":"reassign";L.push([d.id,d.class,dc,f].map(x=>`"${x}"`).join(","))});
const e=document.createElement("a");e.href=URL.createObjectURL(new Blob([L.join("\n")],{type:"text/csv"}));e.download="labels.csv";e.click()}</script>'''
(UAE/a.out).write_text(H.replace("__D__",json.dumps(json.loads(c[["id","class","lat","lon","acq_date","src_res","blur_flag","res_flag"]].to_json(orient="records")))).replace("__C__",json.dumps(CLASSES)).replace("__B__","true" if a.blind else "false"),encoding="utf8")
print("wrote",UAE/a.out,len(c),"candidates")