"""Stricter mimicry: route, amount, lag AND the sender/recipient pattern copied from one benign flow, with fresh addresses of zero history.
Reference: the full benign record copied (exchangeable when paced), which should sit at chance."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
SEEDS=range(20)
def pert(te,y,kind,rng):
    d=te.copy(); pos=np.where(y==1)[0]; ben=np.where(y==0)[0]; idx=d.index[pos]
    pick=rng.choice(ben,size=len(pos),replace=True)
    for c in("src_chain","dst_chain","src_amount","dst_amount","lag_seconds"): d.loc[idx,c]=d[c].to_numpy()[pick]
    if kind.startswith("pattern"):
        ss=te["src_sender"].to_numpy()[pick]; rr=te["dst_recipient"].to_numpy()[pick]
        S=[];R=[]
        for j,(a,b) in enumerate(zip(ss,rr)):
            a=str(a).strip().lower(); b=str(b).strip().lower(); f=f"0xfresh{pos[j]}"
            S.append(f if a else ""); R.append((f if a==b else f+"r") if b else "")
        d.loc[idx,"src_sender"]=S; d.loc[idx,"dst_recipient"]=R; d.loc[idx,"src_recipient"]=""; d.loc[idx,"dst_sender"]=""
    if kind.startswith("full"):
        for c in("src_sender","dst_recipient","src_recipient","dst_sender"): d.loc[idx,c]=d[c].to_numpy()[pick]
    t=d["t"].astype(int).to_numpy().copy()
    if kind.endswith("paced"): t[pos]=rng.integers(t.min(),t.max()+1,size=len(pos))
    d["t"]=t; o=np.argsort(t,kind="stable"); return d.iloc[o].reset_index(drop=True),y[o]
def m(a,y):
    a=np.array(a,dtype=int); tp=int(y[a].sum()) if len(a) else 0; return tp,len(a)-tp
sm=lambda L:dict(tp_mean=round(float(np.mean([x[0] for x in L])),1),tp_min=int(min(x[0] for x in L)),tp_max=int(max(x[0] for x in L)),fp_mean=round(float(np.mean([x[1] for x in L])),1),fp_min=int(min(x[1] for x in L)),fp_max=int(max(x[1] for x in L)))
res={}
for st in FOLDS:
    tr=library(st); te,y=stream_df(full(st)); S=Scorer("extra_trees",tr); rec={}
    for kind in("pattern","pattern_paced","full","full_paced"):
        R=[];E=[]
        for sd in SEEDS:
            d,yy=pert(te,y,kind,np.random.default_rng(3000+sd)); s=S(d); R.append(m(np.where(online_rank(s)>=.999)[0],yy))
            if st==NO: E.append(m(spot(s) or [],yy))
        rec[kind]=dict(rank=sm(R))
        if st==NO: rec[kind]["evt"]=sm(E)
        print(NAMES[st],kind,json.dumps(rec[kind]),flush=True)
    res[NAMES[st]]=rec
# no-exploit window feature content
q,_=stream_df(full(QC).query("flow_type=='single_side_source'"))
print('QC one-record: amount present',int((q.src_amount.str.strip()!="").sum()),'lag present',int((q.lag_seconds.str.strip()!="").sum()),'sender present',int((q.src_sender.str.strip()!="").sum()),'recipient present',int((q.dst_recipient.str.strip()!="").sum()),'chains',q.src_chain.value_counts().to_dict(),'distinct senders',q.src_sender.str.lower().nunique())
res["control_features"]=dict(n=len(q),amount=int((q.src_amount.str.strip()!="").sum()),lag=int((q.lag_seconds.str.strip()!="").sum()),chains=q.src_chain.value_counts().to_dict())
json.dump(res,open(sys.argv[1]+"/step10.json","w"),indent=1); print("DONE")
