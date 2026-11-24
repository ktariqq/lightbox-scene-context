import sys, json, numpy as np, pandas as pd, keras, tensorflow as tf, matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import pointbiserialr
from matplotlib.colors import LinearSegmentedColormap
from config import *
from common import seed_all, cdir, vpath, deco
tf.config.set_visible_devices([],"GPU"); seed_all(); RES.mkdir(exist_ok=True)
mode=sys.argv[1] if len(sys.argv)>1 else "all"; GN=sys.argv[2] if len(sys.argv)>2 else "b4vd"; GS=int(sys.argv[3]) if len(sys.argv)>3 else 42; K=len(CLASSES); SEEDS=(42,7,123)
AR=set(json.load(open(ROOT/"classes.json"))["arid"]); grp=lambda c:"arid" if c in AR else "other"
def ld(sp): ix=pd.read_csv(FEAT/f"{sp}_index.csv"); return {v:np.load(FEAT/f"{v}_{sp}.npy") for v in VIEWS},ix["label"].values,ix["filename"].values
DS={"test":ld("test")}
if (FEAT/"uae_index.csv").exists(): DS["uae"]=ld("uae")
def js(o): return o.item() if hasattr(o,"item") else str(o)
def run(name,s,ds): X,y,_=DS[ds]; return keras.models.load_model(MODELS/f"{name}_s{s}.keras",compile=False).predict({**X,"mask":np.ones((len(y),3),"float32")},batch_size=256,verbose=0)  # [logits, gates (N,3,256), per-view gate (N,3)]
def gates():
    SM,CO=[],[]
    for ds in DS:
        y,fn=DS[ds][1],DS[ds][2]
        for name in ("b4","b4vd"):
            for s in SEEDS:
                lg,_,vg=run(name,s,ds); p=lg.argmax(1); ok=(p==y).astype(float)
                d=pd.DataFrame({"ds":ds,"model":name,"seed":s,"filename":fn,"true":[CLASSES[i] for i in y],"pred":[CLASSES[i] for i in p],"correct":ok,**{f"g_{v}":vg[:,i] for i,v in enumerate(VIEWS)}}); d["group"]=d["true"].map(grp); SM.append(d)
                for v in VIEWS: r,pv=pointbiserialr(ok,d[f"g_{v}"]) if 0<ok.sum()<len(ok) else (np.nan,np.nan); CO.append({"ds":ds,"model":name,"seed":s,"view":v,"r_pointbiserial":r,"p_value":pv})  # gate weight vs correctness
    sm=pd.concat(SM); sm.to_csv(RES/"gates_samples.csv",index=False); pd.DataFrame(CO).to_csv(RES/"gates_corr.csv",index=False)
    L=sm.melt(id_vars=["ds","model","seed","true","group"],value_vars=[f"g_{v}" for v in VIEWS],var_name="view",value_name="gate"); L["view"]=L["view"].str[2:]
    pc=L.groupby(["ds","model","seed","true","group","view"],as_index=False)["gate"].mean()
    pc.groupby(["ds","model","true","group","view"])["gate"].agg(["mean","std"]).reset_index().to_csv(RES/"gates_per_class.csv",index=False)  # mean gate per view per class, mean/std over seeds
    pc.groupby(["ds","model","group","view"])["gate"].agg(["mean","std"]).reset_index().to_csv(RES/"gates_group.csv",index=False)  # arid vs other
    d=sm[(sm["ds"]=="test")&(sm["model"]=="b4vd")&(sm["seed"]==42)]; fig,ax=plt.subplots(1,3,figsize=(12,3.6)); deco(fig)
    for a,v in zip(ax,VIEWS):
        for g,c in (("arid",PURPLE),("other","#6ec1e4")): a.hist(d[d["group"]==g][f"g_{v}"],bins=30,alpha=.7,color=c,label=g)
        a.set(title=v,xlabel="mean gate")
    ax[0].legend(); fig.savefig(RES/"gate_dist.png",dpi=150)
bb=keras.applications.MobileNetV2((224,224,3),include_top=False,weights="imagenet"); fm=keras.Model(bb.input,bb.get_layer("out_relu").output); pre=keras.applications.mobilenet_v2.preprocess_input
CM=LinearSegmentedColormap.from_list("rb",[NAVY,PURPLE,"#f2c14e","#ffffff"]); chk=[True]
def cam(m,xf,v,img,c):
    A=fm(pre(tf.cast(img[None],tf.float32)),training=False)  # backbone to out_relu for one view
    if chk[0]: print("max |GAP(out_relu)-cached feature|:",float(np.abs(tf.reduce_mean(A,[1,2]).numpy()-xf[v]).max())); chk[0]=False
    with tf.GradientTape() as t:
        t.watch(A); o=m({**{u:tf.constant(f) for u,f in xf.items() if u!=v},v:tf.reduce_mean(A,[1,2]),"mask":tf.ones((1,3))},training=False)  # GAP of the map replaces this view; other views stay cached
        s=(o[0] if isinstance(o,list) else o)[0,c]
    g=t.gradient(s,A); h=tf.nn.relu(tf.reduce_sum(tf.reduce_mean(g,[1,2],keepdims=True)*A,-1))[0].numpy(); h=(h/(h.max()+1e-8)).astype("float32")
    return np.clip(np.asarray(Image.fromarray(h).resize((224,224),Image.Resampling.BICUBIC)),0,1)
def ov(img,h): a=(.6*h)[...,None]; return Image.fromarray((img*(1-a)+CM(h)[...,:3]*255*a).astype("uint8"))
def gallery():
    X,y,fn=DS["test"]; m=keras.models.load_model(MODELS/f"{GN}_s{GS}.keras",compile=False); o=m.predict({**X,"mask":np.ones((len(y),3),"float32")},batch_size=256,verbose=0)
    lg=o[0] if isinstance(o,list) else o; vg=o[2] if isinstance(o,list) else np.full((len(y),3),np.nan); e=np.exp(lg-lg.max(1,keepdims=True)); cf=(e/e.sum(1,keepdims=True)).max(1); p=lg.argmax(1)
    rng=np.random.default_rng(SEED); sel=[]  # 3 correct + 2 wrong per class (fewer if the class lacks them), fixed seed
    for c in range(K):
        for kind,n,ix in (("correct",3,np.where((y==c)&(p==c))[0]),("wrong",2,np.where((y==c)&(p!=c))[0])): sel+=[(int(i),kind) for i in rng.choice(ix,min(n,len(ix)),replace=False)]
    G,PM=[],{}
    for i,kind in sel:
        c=CLASSES[y[i]]; stem=fn[i][:-4]; xf={u:X[u][i:i+1] for u in VIEWS}; ent={"filename":fn[i],"class":c,"pred":CLASSES[p[i]],"kind":kind,"correct":bool(p[i]==y[i]),"confidence":float(cf[i]),"gates":vg[i].tolist(),"views":{}}
        for v in VIEWS:
            img=np.asarray(Image.open(vpath(v,"test",c,fn[i])).convert("RGB")); out=RES/"gradcam"/cdir(c)/f"{stem}_{v}.png"; out.parent.mkdir(parents=True,exist_ok=True)
            ov(img,cam(m,xf,v,img,p[i])).save(out); ent["views"][v]={"image":vpath(v,"test",c,fn[i]).relative_to(ROOT).as_posix(),"cam":out.relative_to(ROOT).as_posix()}
        G.append(ent); PM[fn[i]]={v:ent["views"][v]["cam"] for v in VIEWS}
    open(RES/"gallery.json","w").write(json.dumps({"model":GN,"seed":GS,"target":"predicted class logit","items":G},indent=1,default=js).replace("NaN","null")); json.dump(PM,open(RES/"gradcam_paths.json","w"),indent=1); print(len(G),"gallery items")
if mode in ("gates","all"): gates()
if mode in ("gallery","all"): gallery()