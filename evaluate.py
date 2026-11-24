import os, json, time, numpy as np, pandas as pd, keras, tensorflow as tf, matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from config import *
from common import seed_all, cdir, vpath, deco
tf.config.set_visible_devices([],"GPU"); seed_all(); RES.mkdir(exist_ok=True)
K=len(CLASSES); HEADS=["b2","b3","b3vd","b4","b4vd"]; ALL=["customcnn"]+HEADS; SEEDS=(42,7,123); REF=("customcnn","b2"); M=["acc","top3","precision","recall","f1","roc_auc","pr_auc","ece"]
SUB={"".join("csg"[i] for i in range(3) if (c>>i)&1):np.array([(c>>i)&1 for i in range(3)],"float32") for c in range(1,8)}  # subset name -> mask, bit i = view i (c, s, g)
AR=set(json.load(open(ROOT/"classes.json"))["arid"]); grp=lambda c:"arid" if c in AR else "other"
if not AR: print("WARNING: arid list empty, group tables will be empty")
def ld(sp): ix=pd.read_csv(FEAT/f"{sp}_index.csv"); return {v:np.load(FEAT/f"{v}_{sp}.npy") for v in VIEWS},ix["label"].values,ix["filename"].values
DS={"test":ld("test")}
if (FEAT/"uae_index.csv").exists(): DS["uae"]=ld("uae")
else: print("WARNING: no UAE features, UAE skipped")
if (FEAT/"meet10_index.csv").exists(): DS["meet10"]=ld("meet10")  # MEET test degraded to 10 m (resolution control)
ytr=pd.read_csv(FEAT/"train_index.csv")["label"].values; MC={}
def MODEL(n,s):
    if (n,s) not in MC: MC[n,s]=keras.models.load_model(MODELS/("customcnn.keras" if n=="customcnn" else f"{n}_s{s}.keras"),compile=False)
    return MC[n,s]
def sm(z): e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
def pth(ds):
    if ds=="test": d=pd.read_csv(MAN); d=d[d["split"]=="test"]; return [str(vpath("center","test",c,f)) for f,c in zip(d["filename"],d["class"])]
    if ds=="meet10": d=pd.read_csv(MAN); d=d[d["split"]=="test"]; return [str(MEET10/"center"/cdir(c)/(Path(f).stem+".jpg")) for f,c in zip(d["filename"],d["class"])]
    u=pd.read_csv(UMAN); return [str(UAE_SUBSET/"center"/cdir(c)/f"{i}.jpg") for i,c in zip(u["id"],u["class"])]
def lb1(ds): return MODEL("customcnn",42).predict(tf.data.Dataset.from_tensor_slices(pth(ds)).map(lambda p:tf.ensure_shape(tf.cast(tf.io.decode_jpeg(tf.io.read_file(p),channels=3),tf.float32),[224,224,3])).batch(64),verbose=0)
def lg(n,s,ds,sub):
    X,y,_=DS[ds]; m=MODEL(n,s)
    if n=="b2": return m.predict({"center":X["center"]},batch_size=256,verbose=0)
    o=m.predict({**X,"mask":np.tile(SUB[sub],(len(y),1))},batch_size=256,verbose=0); return o[0] if isinstance(o,list) else o
def met(y,z,pres,rs=False):  # macro metrics over the classes evaluated; rs=True restricts logits to the present classes
    pres=np.array(pres)
    if rs: z=z[:,pres]; y=np.searchsorted(pres,y); cl=np.arange(len(pres))
    else: cl=pres
    P=sm(z); p=P.argmax(1); pr,rc,f1,sp=precision_recall_fscore_support(y,p,labels=cl,zero_division=0)
    ra=[roc_auc_score(y==c,P[:,c]) if 0<(y==c).sum()<len(y) else np.nan for c in cl]; pa=[average_precision_score(y==c,P[:,c]) for c in cl]
    cf=P.max(1); ok=p==y; b=np.minimum((cf*15).astype(int),14); ece=sum(abs(ok[b==i].mean()-cf[b==i].mean())*(b==i).mean() for i in range(15) if (b==i).any())  # ece: 15 equal-width bins
    a={"acc":ok.mean(),"top3":(np.argsort(-P,1)[:,:3]==y[:,None]).any(1).mean(),"precision":pr.mean(),"recall":rc.mean(),"f1":f1.mean(),"roc_auc":np.nanmean(ra),"pr_auc":np.nanmean(pa),"ece":ece}
    return a,pd.DataFrame({"class":[CLASSES[c] for c in pres],"precision":pr,"recall":rc,"f1":f1,"support":sp,"roc_auc":ra,"pr_auc":pa})
RUN,PC,ERR,LG=[],[],[],{}
def add(ds,n,s,sub,z):
    y,fn=DS[ds][1],DS[ds][2]; pres=sorted(set(y)); LG[ds,n,s,sub]=z
    for var in (("full24","restricted") if ds=="uae" else ("full24",)):
        a,pc=met(y,z,pres,var=="restricted"); RUN.append({"ds":ds,"variant":var,"model":n,"seed":s,"subset":sub,**a})
        if sub in ("csg","all"): PC.append(pc.assign(ds=ds,variant=var,model=n,seed=s,group=pc["class"].map(grp)))
    if sub in ("csg","all") and n not in ("majority","chance"):
        P=sm(z); p=P.argmax(1); w=np.where(p!=y)[0]; ERR.append(pd.DataFrame({"ds":ds,"model":n,"seed":s,"filename":fn[w],"true":[CLASSES[i] for i in y[w]],"pred":[CLASSES[i] for i in p[w]],"confidence":P.max(1)[w]}))
for ds in DS:
    n=len(DS[ds][1])
    add(ds,"majority",42,"all",np.tile(np.log(np.bincount(ytr,minlength=K)/len(ytr)+1e-12),(n,1)))  # train-prior argmax (ties -> first class)
    add(ds,"chance",42,"all",np.random.default_rng(SEED).normal(size=(n,K))); add(ds,"customcnn",42,"all",lb1(ds))
    for s in SEEDS:
        add(ds,"b2",s,"all",lg("b2",s,ds,"c"))
        for h in HEADS[1:]:
            for sub in SUB: add(ds,h,s,sub,lg(h,s,ds,sub))
runs=pd.DataFrame(RUN); runs.to_csv(RES/"runs.csv",index=False)
def tab(d): g=d.groupby("model",sort=False)[M].agg(["mean","std"]); g.columns=[f"{a}_{b}" for a,b in g.columns]; g["n_seeds"]=d.groupby("model",sort=False).size(); return g.reset_index()
for (ds,v),d in runs.groupby(["ds","variant"]):
    for sub in SUB: tab(d[d["subset"].isin([sub,"all"])]).to_csv(RES/f"table_{ds}_{v}_{sub}.csv",index=False)
pc=pd.concat(PC); pc.to_csv(RES/"per_class.csv",index=False); pc[pc["group"]=="arid"].to_csv(RES/"per_class_arid.csv",index=False); pc[pc["group"]=="other"].to_csv(RES/"per_class_other.csv",index=False)
gs=pc.groupby(["ds","variant","model","group"],sort=False)[["precision","recall","f1"]].mean().reset_index(); gs.to_csv(RES/"group_summary.csv",index=False)
pd.concat(ERR).to_csv(RES/"errors.csv",index=False); TC=[]
for ds in DS:
    y=DS[ds][1]
    for n in ALL:
        z=LG[ds,n,42,"all" if n in REF else "csg"]; cm=confusion_matrix(y,z.argmax(1),labels=range(K)); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(RES/f"confusion_{ds}_{n}.csv")
        o=cm.copy(); np.fill_diagonal(o,0)
        for i,j in zip(*np.unravel_index(np.argsort(-o,axis=None)[:10],o.shape)):
            if o[i,j]>0: TC.append({"ds":ds,"model":n,"true":CLASSES[i],"pred":CLASSES[j],"count":int(o[i,j]),"rate_of_true_class":float(o[i,j]/cm[i].sum())})
pd.DataFrame(TC).to_csv(RES/"top_confusions.csv",index=False)
y=DS["test"][1]; N=len(y); pp=lambda n,s:LG["test",n,s,"all" if n in REF else "csg"].argmax(1); MN=[]
for a,b in [("b2","b3"),("b3","b4"),("b3vd","b4vd"),("b3","b3vd"),("b4","b4vd")]:  # paired exact McNemar on MEET test, all three views
    for s in SEEDS:
        ca,cb=pp(a,s)==y,pp(b,s)==y; x,w=int((ca&~cb).sum()),int((~ca&cb).sum()); p=binomtest(min(x,w),x+w,.5).pvalue if x+w else 1.0; d=cb.mean()-ca.mean()
        MN.append({"a":a,"b":b,"seed":s,"a_only_correct":x,"b_only_correct":w,"acc_a":ca.mean(),"acc_b":cb.mean(),"delta_acc":d,"odds_ratio":(w+.5)/(x+.5),"p_value":p,"significant":p<.05,"practical":abs(d)>=PRACT})
mn=pd.DataFrame(MN); mn["verdict"]=np.select([mn["significant"]&mn["practical"],mn["significant"],mn["practical"]],["significant+practical","significant only","practical only"],"neither"); mn.to_csv(RES/"mcnemar.csv",index=False)
def pcm(yy,p,I): yb,pb=yy[I],p[I]; tp=np.stack([((yb==c)&(pb==c)).sum(1) for c in range(K)],1); ny=np.stack([(yb==c).sum(1) for c in range(K)],1); npd=np.stack([(pb==c).sum(1) for c in range(K)],1); return 2*tp/np.maximum(ny+npd,1),tp/np.maximum(ny,1)
rng=np.random.default_rng(SEED); ix=[np.where(y==c)[0] for c in range(K)]; I=np.concatenate([rng.choice(i,(NBOOT,len(i))) for i in ix],1); I0=np.arange(N)[None]  # unit: test image, resampled with replacement within its class; same draws for both models (paired)
ar=np.array([c in AR for c in CLASSES]); CG,GG=[],[]
for s in SEEDS:
    f0,a0=pcm(y,pp("b2",s),I0); fb,ab=pcm(y,pp("b2",s),I)
    for h in HEADS[1:]:
        p=pp(h,s); f1,a1=pcm(y,p,I0); fB,aB=pcm(y,p,I); G={}
        for mt,(pt,bt) in {"f1":((f1-f0)[0],fB-fb),"acc":((a1-a0)[0],aB-ab)}.items():  # context gain = fusion minus b2
            lo,hi=np.percentile(bt,[2.5,97.5],axis=0); G[mt]=(pt,lo,hi)
            if ar.any() and (~ar).any(): d=bt[:,ar].mean(1)-bt[:,~ar].mean(1); GG.append({"model":h,"seed":s,"metric":mt,"gain_arid":pt[ar].mean(),"gain_other":pt[~ar].mean(),"diff_arid_minus_other":pt[ar].mean()-pt[~ar].mean(),"diff_lo":np.percentile(d,2.5),"diff_hi":np.percentile(d,97.5)})
        CG+=[{"model":h,"seed":s,"class":c,"group":grp(c),"gain_f1":G["f1"][0][i],"f1_lo":G["f1"][1][i],"f1_hi":G["f1"][2][i],"gain_acc":G["acc"][0][i],"acc_lo":G["acc"][1][i],"acc_hi":G["acc"][2][i]} for i,c in enumerate(CLASSES)]
pd.DataFrame(CG).to_csv(RES/"context_gain.csv",index=False); pd.DataFrame(GG).to_csv(RES/"context_gain_groups.csv",index=False)
bb=keras.applications.MobileNetV2((224,224,3),include_top=False,weights="imagenet",pooling="avg"); bb.save(MODELS/"mobilenetv2_backbone.keras")
TT=json.load(open(RES/"train_times.json")); FM=json.load(open(FEAT/"features_meta.json")); im=tf.random.uniform((1,224,224,3),0,255,seed=1); ft=tf.random.uniform((1,1280),seed=2)
def lat(f,n=100,w=10):  # cpu, batch size 1, after warm-up, mean over n runs
    for _ in range(w): f()
    t=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t)/n*1000
EF=[]
for n in ALL:
    m=MODEL(n,42); nv=1 if n=="b2" else 3; mk=lambda f:{"center":f[0]} if n=="b2" else {**dict(zip(VIEWS,f)),"mask":tf.ones((1,3))}
    if n=="customcnn": hd=None; e2e=lat(lambda:m(im,training=False))
    else: hd=lat(lambda:m(mk([ft]*nv),training=False)); e2e=lat(lambda:m(mk([bb(im,training=False) for _ in range(nv)]),training=False))
    sec=[v for k,v in TT.items() if k==n or k.startswith(n+"_s")]
    EF.append({"model":n,"trainable_params":int(sum(np.prod(w.shape) for w in m.trainable_weights)),"total_params":int(m.count_params()),"backbone_params":0 if n=="customcnn" else int(bb.count_params()),"train_seconds_mean":float(np.mean(sec)),"size_mb":os.path.getsize(MODELS/("customcnn.keras" if n=="customcnn" else f"{n}_s42.keras"))/1e6,"latency_head_ms":hd,"latency_e2e_ms":e2e})
pd.DataFrame(EF).to_csv(RES/"efficiency.csv",index=False)
FE={sp:{**v,"ms_per_image_per_view":round(v["seconds"]/(v["n"]*3)*1000,3)} for sp,v in FM.items()}; json.dump(FE,open(RES/"features_timing.json","w"),indent=1)
def crv(yy,P):
    g=np.linspace(0,1,101); T,Q=[],[]
    for c in range(K): f,t,_=roc_curve(yy==c,P[:,c]); T.append(np.interp(g,f,t)); a,r,_=precision_recall_curve(yy==c,P[:,c]); Q.append(np.interp(g,r[::-1],a[::-1]))
    return g,np.mean(T,0),np.mean(Q,0)
fig,ax=plt.subplots(1,2,figsize=(11,4.5)); deco(fig); fr,fa=plt.subplots(figsize=(5.5,5)); deco(fr); fa.plot([0,1],[0,1],ls="--",c="#8a7bb0"); CV={}
for n in ALL:
    P=sm(LG["test",n,42,"all" if n in REF else "csg"]); g,t,q=crv(y,P); CV[n]={"grid":g.tolist(),"tpr":t.tolist(),"precision":q.tolist()}; kw=dict(c=PURPLE,lw=2.6) if n=="b4vd" else dict(lw=1.2)
    ax[0].plot(g,t,label=n,**kw); ax[1].plot(g,q,label=n,**kw); b=np.minimum((P.max(1)*10).astype(int),9); ok=P.argmax(1)==y; cb=[i for i in range(10) if (b==i).any()]
    fa.plot([(i+.5)/10 for i in cb],[ok[b==i].mean() for i in cb],marker="o",label=n,**kw)
ax[0].set(title="macro one-vs-rest ROC (test)",xlabel="FPR",ylabel="TPR"); ax[1].set(title="macro one-vs-rest PR (test)",xlabel="recall",ylabel="precision"); ax[0].legend(); fa.set(title="reliability (test, 10 bins)",xlabel="confidence",ylabel="accuracy"); fa.legend()
fig.savefig(RES/"roc_pr_test.png",dpi=150); fr.savefig(RES/"reliability_test.png",dpi=150); json.dump(CV,open(RES/"curves.json","w"))
def js(o): return o.item() if hasattr(o,"item") else str(o)
OUT={"seeds":list(SEEDS),"subsets":list(SUB),"arid":sorted(AR),"main_tables":{f"{ds}_{v}":tab(d[d["subset"].isin(["csg","all"])]).set_index("model").to_dict("index") for (ds,v),d in runs.groupby(["ds","variant"])},
 "group_summary":gs.to_dict("records"),"mcnemar":mn.to_dict("records"),"context_gain_groups":GG,"efficiency":EF,"features_timing":FE,
 "uae":{"n":int(len(DS["uae"][1])),"counts":{CLASSES[c]:int((DS["uae"][1]==c).sum()) for c in sorted(set(DS["uae"][1]))}} if "uae" in DS else None,
 "notes":{"bootstrap":f"unit = test image, resampled with replacement within its class (stratified), same draws for both models (paired), {NBOOT} draws, percentile 95% CI","paired_tests":"exact McNemar on MEET test, all three views, per seed","practical_threshold_acc":PRACT,"b1_vs_b2":"architectures differ, not a context comparison","uae":"24-way models; macro metrics over present classes (full24); restricted variant uses logits of present classes only","baselines":"majority = train-prior argmax; chance = random logits seed 42"}}
open(RES/"metrics.json","w").write(json.dumps(OUT,indent=1,default=js).replace("NaN","null")); print("done")