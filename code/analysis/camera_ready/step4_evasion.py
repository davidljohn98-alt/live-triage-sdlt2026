"""Evasion replay: attacker flows perturbed in the test stream, library and model frozen, every policy parameter unchanged."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
SEEDS=range(20)
def pert(te,y,kind,rng):
    d=te.copy(); pos=np.where(y==1)[0]; ben=np.where(y==0)[0]
    pick=rng.choice(ben,size=len(pos),replace=True)
    if kind in("route","all"):
        d.loc[d.index[pos],"src_chain"]=d["src_chain"].to_numpy()[pick]; d.loc[d.index[pos],"dst_chain"]=d["dst_chain"].to_numpy()[pick]
    if kind in("amount","all"):
        pick2=rng.choice(ben,size=len(pos),replace=True)
        for c in("src_amount","dst_amount"): d.loc[d.index[pos],c]=d[c].to_numpy()[pick2]
    if kind in("lag","all"):
        pick3=rng.choice(ben,size=len(pos),replace=True); d.loc[d.index[pos],"lag_seconds"]=d["lag_seconds"].to_numpy()[pick3]
    if kind in("address","all"):
        d.loc[d.index[pos],"src_sender"]=[f"0xfresh{i}s" if str(v).strip() else "" for i,v in zip(pos,d["src_sender"].to_numpy()[pos])]
        d.loc[d.index[pos],"dst_recipient"]=[f"0xfresh{i}r" if str(v).strip() else "" for i,v in zip(pos,d["dst_recipient"].to_numpy()[pos])]
        d.loc[d.index[pos],"src_recipient"]=""; d.loc[d.index[pos],"dst_sender"]=""
    t=d["t"].astype(int).to_numpy().copy()
    if kind in("pacing","all"):
        t[pos]=rng.integers(t.min(),t.max()+1,size=len(pos))
    d["t"]=t
    o=np.argsort(t,kind="stable"); d=d.iloc[o].reset_index(drop=True); yy=y[o]
    return d,yy
def m(a,y):
    a=np.array(a,dtype=int); tp=int(y[a].sum()) if len(a) else 0; return tp,len(a)-tp
def propag(te,y,seedidx):
    S=set(addrs(te.iloc[seedidx])); t=te["t"].astype(int).to_numpy(); tp=fp=0
    for i in range(seedidx+1,len(te)):
        A=addrs(te.iloc[i])
        if A&S:
            tp+=int(y[i]==1); fp+=int(y[i]==0); S|=A
    return tp,fp
res={}
for st in FOLDS:
    tr=library(st); te,y=stream_df(full(st)); S=Scorer("extra_trees",tr)
    base=S(te); rk=online_rank(base); aB=np.where(rk>=.999)[0]
    rec=dict(npos=int(y.sum()),base_rank=m(aB,y))
    amt=amount_scores(te); rec["base_amount1"]=m(np.where(online_rank(amt)>=.99)[0],y)
    if st==NO: rec["base_evt"]=m(spot(base),y)
    for kind in("route","amount","lag","pacing","address","all"):
        R=[];E=[];Am=[];PP=[]
        for sd in SEEDS:
            rng=np.random.default_rng(1000+sd); d,yy=pert(te,y,kind,rng)
            s=S(d); r=online_rank(s); R.append(m(np.where(r>=.999)[0],yy))
            if st==NO:
                al=spot(s) or []; E.append(m(al,yy))
                la=[i for i in al if yy[i]==1]
                if la: PP.append(propag(d,yy,la[0]))
                else: PP.append((0,0))
            ra=online_rank(amount_scores(d)); Am.append(m(np.where(ra>=.99)[0],yy))
        sm=lambda L:dict(tp_mean=round(float(np.mean([x[0] for x in L])),1),tp_min=int(min(x[0] for x in L)),tp_max=int(max(x[0] for x in L)),fp_mean=round(float(np.mean([x[1] for x in L])),1),fp_min=int(min(x[1] for x in L)),fp_max=int(max(x[1] for x in L)))
        rec[kind]=dict(rank=sm(R),amount1=sm(Am))
        if st==NO: rec[kind]["evt"]=sm(E); rec[kind]["prop_from_first_evt"]=sm(PP)
        print(NAMES[st],kind,json.dumps(rec[kind]),flush=True)
    res[NAMES[st]]=rec
json.dump(res,open(sys.argv[1]+"/step4.json","w"),indent=1); print("DONE")
