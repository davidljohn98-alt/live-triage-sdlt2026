"""Joint mimicry and single-attribute evasion, fixed semantics: route, amount and lag copied from the SAME benign flow in the joint variants."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
SEEDS=range(20)
def pert(te,y,kind,rng):
    d=te.copy(); pos=np.where(y==1)[0]; ben=np.where(y==0)[0]; idx=d.index[pos]
    pick=rng.choice(ben,size=len(pos),replace=True)
    def copy(cols):
        for c in cols: d.loc[idx,c]=d[c].to_numpy()[pick]
    if kind=="route": copy(("src_chain","dst_chain"))
    if kind=="amount": copy(("src_amount","dst_amount"))
    if kind=="lag": copy(("lag_seconds",))
    if kind in("mimic","mimic_paced"): copy(("src_chain","dst_chain","src_amount","dst_amount","lag_seconds"))
    if kind in("address","mimic","mimic_paced"):
        d.loc[idx,"src_sender"]=[f"0xfresh{i}s" for i in pos]; d.loc[idx,"dst_recipient"]=[f"0xfresh{i}r" for i in pos]; d.loc[idx,"src_recipient"]=""; d.loc[idx,"dst_sender"]=""
    t=d["t"].astype(int).to_numpy().copy()
    if kind in("pacing","mimic_paced"): t[pos]=rng.integers(t.min(),t.max()+1,size=len(pos))
    d["t"]=t; o=np.argsort(t,kind="stable"); return d.iloc[o].reset_index(drop=True),y[o]
def m(a,y):
    a=np.array(a,dtype=int); tp=int(y[a].sum()) if len(a) else 0; return tp,len(a)-tp
sm=lambda L:dict(tp_mean=round(float(np.mean([x[0] for x in L])),1),tp_min=int(min(x[0] for x in L)),tp_max=int(max(x[0] for x in L)),fp_mean=round(float(np.mean([x[1] for x in L])),1),fp_min=int(min(x[1] for x in L)),fp_max=int(max(x[1] for x in L)))
res={}
for st in FOLDS:
    tr=library(st); te,y=stream_df(full(st)); S=Scorer("extra_trees",tr); rec={"npos":int(y.sum()),"n":len(te)}
    for kind in("route","amount","lag","pacing","address","mimic","mimic_paced"):
        R=[];E=[]
        for sd in SEEDS:
            d,yy=pert(te,y,kind,np.random.default_rng(2000+sd)); s=S(d); R.append(m(np.where(online_rank(s)>=.999)[0],yy))
            if st==NO: E.append(m(spot(s) or [],yy))
        rec[kind]=dict(rank=sm(R))
        if st==NO: rec[kind]["evt"]=sm(E)
        print(NAMES[st],kind,json.dumps(rec[kind]),flush=True)
    res[NAMES[st]]=rec
json.dump(res,open(sys.argv[1]+"/step7.json","w"),indent=1); print("DONE")
