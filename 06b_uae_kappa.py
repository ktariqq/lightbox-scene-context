import sys, pandas as pd
from sklearn.metrics import cohen_kappa_score
from config import *
if sys.argv[1]=="sample":
    l=pd.read_csv(UAE/"labels.csv"); l=l[l["final_class"].notna()]; l["id"].sample(min(50,len(l)),random_state=SEED).to_csv(UAE/"sample50.csv",index=False)  # random 50 reviewed ids
else:
    a,b=[pd.read_csv(p).set_index("id")["final_class"].dropna() for p in sys.argv[2:4]]; j=a.index.intersection(b.index)
    print("n",len(j),"agreement",(a[j]==b[j]).mean().round(3),"kappa",round(cohen_kappa_score(a[j],b[j]),3))  # reject counts as its own label