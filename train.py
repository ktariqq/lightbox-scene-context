import sys, json, time, math, numpy as np, pandas as pd, keras, keras_tuner as kt, tensorflow as tf
from sklearn.metrics import f1_score
from config import *
from common import seed_all, cdir
from models import customcnn, transfercnn, fusionhead, NC
EP_H,PAT_H,SEEDS=30,5,(42,7,123); ARCH={"b3":"concat","b3vd":"concat","b4":"gated","b4vd":"gated"}; KEYS=VIEWS+["mask"]
DEF={"lr":1e-3,"proj":256,"hid":128,"pfull":0.5,"bs":64}  # EDIT: default config used by "heads default"
SCE=keras.losses.SparseCategoricalCrossentropy(from_logits=True); st=sys.argv[1] if len(sys.argv)>1 else "all"
seed_all(); MODELS.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
def rj(n): p=RES/n; return json.load(open(p)) if p.exists() else {}
def wj(n,o): json.dump(o,open(RES/n,"w"),indent=1)
TIMES,HIST=rj("train_times.json"),rj("history.json"); mm=pd.read_csv(MAN); ymn=mm[mm["split"]=="train"]["class"].map(CLASSES.index).values
def cw(y): c=np.bincount(y,minlength=NC); return None if len(set(c))==1 else {i:float(len(y)/(NC*c[i])) for i in range(NC)}  # class weights only if counts are uneven
CW=cw(ymn); print("class weights:",CW or "none (even counts)")
def load(sp): ix=pd.read_csv(FEAT/f"{sp}_index.csv"); return {v:np.load(FEAT/f"{v}_{sp}.npy") for v in VIEWS},ix["label"].values.astype("int32")
if st!="b1": Xtr,ytr=load("train"); Xva,yva=load("val"); VA={**Xva,"mask":np.ones((len(yva),3),"float32")}
def rmask(x,y,p):
    k=tf.random.uniform([],0,6,tf.int32)+1; c=tf.where(tf.random.uniform([])<p,7,k)  # per-sample: full views w.p. p, else uniform over the 6 other non-empty subsets (codes 1..6)
    x=dict(x); x["mask"]=tf.cast(tf.bitwise.bitwise_and(tf.bitwise.right_shift(c,tf.range(3)),1),tf.float32); return x,y
def mkds(X,y,bs,keys,train=False,pf=1.0,seed=SEED):
    ds=tf.data.Dataset.from_tensor_slices(({k:(X[k] if k in X else np.ones((len(y),3),"float32")) for k in keys},y))
    if train: ds=ds.shuffle(len(y),seed=seed)
    if train and "mask" in keys: ds=ds.map(lambda x,t:rmask(x,t,pf))
    return ds.batch(bs).prefetch(1)
def train(tm,c,seed,keys,pf):
    spe=math.ceil(len(ytr)/c["bs"]); tm.compile(keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(c["lr"],spe*EP_H)),loss=SCE,metrics=["accuracy"])  # decay_steps = steps_per_epoch*max_epochs, independent of early stopping
    t=time.time(); h=tm.fit(mkds(Xtr,ytr,c["bs"],keys,True,pf,seed),epochs=EP_H,validation_data=mkds(Xva,yva,256,keys),callbacks=[keras.callbacks.EarlyStopping(patience=PAT_H,restore_best_weights=True)],class_weight=CW,verbose=0)
    return {k:[float(v) for v in vs] for k,vs in h.history.items()},time.time()-t
def space(hp): return {"lr":hp.Choice("lr",[3e-4,1e-3,3e-3]),"proj":hp.Choice("proj",[128,256,512]),"hid":hp.Choice("hid",[64,128,256]),"pfull":hp.Choice("pfull",[0.3,0.5,0.7]),"bs":hp.Choice("bs",[32,64,128])}  # EDIT: search space values
def hm(arch):
    def build(hp): c=space(hp); return fusionhead(arch,c["proj"],c["hid"])[0]
    return build
class f1tuner(kt.RandomSearch):
    def run_trial(self,trial):
        hp=trial.hyperparameters; c=space(hp); seed_all(SEED); tm=self.hypermodel.build(hp); train(tm,c,SEED,KEYS,c["pfull"])
        f=float(f1_score(yva,tm.predict(VA,batch_size=256,verbose=0).argmax(1),average="macro",labels=list(range(NC)),zero_division=0)); keras.backend.clear_session(); return f  # macro-F1 on the whole val set, all 3 views
def tune():
    T={}
    for name in ("b3vd","b4vd"):
        t=f1tuner(hypermodel=hm(ARCH[name]),objective=kt.Objective("macro_f1","max"),max_trials=10,seed=SEED,directory=str(RES/"kt"),project_name=name,overwrite=True); t.search()
        T[name]={"best":t.oracle.get_best_trials(1)[0].hyperparameters.values,"trials":[{"id":x.trial_id,"hp":x.hyperparameters.values,"val_macro_f1":x.score} for x in t.oracle.trials.values()]}
    wj("tuning.json",T)
def cfgs():
    if "default" in sys.argv: return {"b2":{**DEF,"pfull":1.0},"b3":{**DEF,"pfull":1.0},"b3vd":DEF,"b4":{**DEF,"pfull":1.0},"b4vd":DEF}
    T=rj("tuning.json"); b3,b4=T["b3vd"]["best"],T["b4vd"]["best"]  # no-VD reuse VD config with pfull=1.0; b2 reuses b3vd lr and sizes
    return {"b2":{"lr":b3["lr"],"proj":b3["proj"],"hid":b3["hid"],"bs":64,"pfull":1.0},"b3":{**b3,"pfull":1.0},"b3vd":b3,"b4":{**b4,"pfull":1.0},"b4vd":b4}
def heads():
    cf=cfgs(); S={}; wj("configs.json",cf)
    for name,c in cf.items():
        r={"val_acc":[],"val_f1":[]}; keys=["center"] if name=="b2" else KEYS
        for s in SEEDS:
            seed_all(s); tm,fm=(lambda m:(m,m))(transfercnn(c["proj"],c["hid"])) if name=="b2" else fusionhead(ARCH[name],c["proj"],c["hid"])
            h,t=train(tm,c,s,keys,c["pfull"]); p=tm.predict({k:VA[k] for k in keys},batch_size=256,verbose=0).argmax(1)  # validation always uses all three views
            r["val_acc"].append(float((p==yva).mean())); r["val_f1"].append(float(f1_score(yva,p,average="macro",zero_division=0)))
            fm.save(MODELS/f"{name}_s{s}.keras"); TIMES[f"{name}_s{s}"]=round(t,1); HIST[f"{name}_s{s}"]=h; wj("train_times.json",TIMES); wj("history.json",HIST); keras.backend.clear_session()
        r["val_f1_mean"],r["val_f1_std"]=float(np.mean(r["val_f1"])),float(np.std(r["val_f1"])); S[name]=r; wj("head_val_summary.json",S); print(name,r)
def b1():
    seed_all(SEED); ld=lambda sp,sh: keras.utils.image_dataset_from_directory(str(SUBSET/"center"/sp),class_names=[cdir(c) for c in CLASSES],image_size=(224,224),batch_size=32,shuffle=sh,seed=SEED).prefetch(tf.data.AUTOTUNE)
    tr,va=ld("train",True),ld("val",False); m=customcnn()
    m.compile(keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3,len(tr)*20)),loss=SCE,metrics=["accuracy"])  # decay_steps = steps_per_epoch*max_epochs
    t=time.time(); h=m.fit(tr,epochs=20,validation_data=va,callbacks=[keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True)],class_weight=CW,verbose=2)
    TIMES["customcnn"]=round(time.time()-t,1); HIST["customcnn"]={k:[float(v) for v in vs] for k,vs in h.history.items()}; m.save(MODELS/"customcnn.keras"); wj("train_times.json",TIMES); wj("history.json",HIST)
if st in ("b1","all"): b1()
if st in ("tune","all"): tune()
if st in ("heads","all"): heads()