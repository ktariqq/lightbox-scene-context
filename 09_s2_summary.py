import json, pandas as pd
from config import *
cov=json.load(open(RES/"uae_coverage.json"))["covered"]; pc=pd.read_csv(RES/"per_class.csv"); r=pd.read_csv(RES/"runs.csv")
pc=pc[(pc["variant"]=="full24")&pc["class"].isin(cov)]; r=r[r["variant"]=="full24"]
a=pc.groupby(["model","ds"])["f1"].mean().unstack().reindex(["majority","customcnn","b2","b3","b3vd","b4","b4vd"]); a.loc["b4vd - b2"]=a.loc["b4vd"]-a.loc["b2"]  # macro-F1 on UAE-covered classes only
b=r[r["model"]=="b4vd"].groupby(["ds","subset"])["f1"].mean().unstack().reindex(columns=["c","s","g","cs","cg","sg","csg"])  # view-subset effect inside each dataset
a.to_csv(RES/"s2_summary_classes.csv"); b.to_csv(RES/"s2_summary_subsets.csv")
print("macro-F1 on UAE-covered classes\n",a.round(3).to_string(),"\n\nB4-VD macro-F1 by view subset\n",b.round(3).to_string())