#!/usr/bin/env python3
"""JobB: (1) permuted-ordering learning curve (does the jump track Meter or size?);
(2) rank grid JSON for appendix; (3) GNN on the negative-control stream."""
import sys, os, bisect, json, math
GEN="<ARTEFACT_ROOT>"
sys.path.insert(0,GEN); sys.path.insert(0,GEN+"/live_detection")
import numpy as np, pandas as pd
import train_models_per_case as TM
import train_score_graph_models as TG
P=GEN+"/data/observed_panels_v2"
CS,PO,MC,QU,ME,HA,NO=("chainswap_2021_authentication_bypass","poly_network_2021_authentication_bypass",
 "multichain_2022_authentication_bypass","qubit_2022_bad_business_logic",
 "meter_io_2022_bad_business_logic","harmony_horizon_2022_private_key_leak",
 "nomad_2022_authentication_bypass")
NAMES={CS:"ChainSwap",PO:"Poly",MC:"Multichain",QU:"Qubit",ME:"Meter",HA:"Harmony",NO:"Nomad"}
def full(s):
    return pd.concat([pd.read_csv(f"{P}/{s}_{sp}.csv",dtype=str,keep_default_na=False) for sp in("train","test")],ignore_index=True)
def t_obs(r):
    ts=[]
    for c in("src_block_timestamp","dst_block_timestamp"):
        v=str(r.get(c,"")).strip()
        if v:
            try: ts.append(int(float(v)))
            except ValueError: pass
    return max(ts) if ts else None
panels={s:full(s) for s in (CS,PO,MC,QU,ME,HA,NO)}
def stream(stem):
    te=panels[stem].copy(); te["t"]=[t_obs(r) for _,r in te.iterrows()]
    te=te[te["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
    y=(te[TM.LABEL_COL].astype(str)=="1").astype(int).to_numpy()
    return te,y
def spot_alerts(s,q=1e-3,init_q=0.98):
    n=len(s); n_init=max(150,int(0.15*n))
    if n<500: return None
    init=s[:n_init]; t=float(np.quantile(init,init_q)); peaks=list(init[init>t]-t)
    def zq(Nt,ncur):
        if len(peaks)<10: return None
        pk=np.array(peaks); m,v=pk.mean(),pk.var()
        if v<=0: return t+m
        g=0.5*(1-m*m/v); sg=0.5*m*(m*m/v+1); r=q*ncur/max(Nt,1)
        try:
            if abs(g)<1e-6: return t-sg*math.log(r)
            return t+(sg/g)*((r**(-g))-1)
        except (OverflowError,ValueError): return None
    z=zq(len(peaks),n_init) or float(np.quantile(init,0.999))
    al=[]; Nt=len(peaks); ncur=n_init
    for i in range(n_init,n):
        x=s[i]; ncur+=1
        if x>z: al.append(i)
        elif x>t:
            peaks.append(x-t); Nt+=1
            z2=zq(Nt,ncur)
            if z2 is not None: z=z2
    return al
def scores_for(mname,tr,te):
    if mname=="learned_graph":
        sm=TG.learned_graph_scores(tr,te.copy())
    else:
        out=TM.fit_score(mname,tr,te.copy())
        sm={str(r["flow_id"]):float(r["model_score"]) for _,r in out.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])
def online_rank(s):
    out=np.zeros(len(s)); seen=[]
    for i,x in enumerate(s):
        out[i]=bisect.bisect_left(seen,x)/len(seen) if seen else 0.0
        bisect.insort(seen,float(x))
    return out

# ---- (1) permuted learning curve ----
print("=== PERMUTED LEARNING CURVE (test=Nomad, EVT q=1e-3, TP linked recovered) ===",flush=True)
te,y=stream(NO); npos=int(y.sum())
ORDERINGS={"chrono":[CS,PO,MC,QU,ME,HA],
           "meter-first":[ME,CS,PO,MC,QU,HA],
           "meter-second":[CS,ME,PO,MC,QU,HA],
           "meter-last":[CS,PO,MC,QU,HA,ME],
           "reversed":[HA,ME,QU,MC,PO,CS]}
res={}
for oname,order in ORDERINGS.items():
    res[oname]={}
    for mname,mlab in (("extra_trees","ET"),("gradient_boosting","GB"),("random_forest","RF")):
        tps=[]
        for k in range(1,7):
            tr=pd.concat([panels[s] for s in order[:k]],ignore_index=True)
            try:
                s=scores_for(mname,tr,te.copy())
                al=spot_alerts(s) or []
                tp=int(sum(1 for i in al if y[i]==1))
            except Exception as e:
                tp=-1
            tps.append(tp)
        res[oname][mlab]=tps
        print(f"  {oname:13s} {mlab}: sizes1-6 -> {tps}   [order: {'>'.join(NAMES[s] for s in order)}]",flush=True)
json.dump(res,open(GEN+"/live_detection/permuted_learning_curve.json","w"),indent=1)

# ---- (2) rank grid JSON (7 models x 4 folds) ----
print("=== RANK GRID JSON ===",flush=True)
ORDER=[CS,PO,MC,QU,ME,HA,NO]
grid=[]
for k,stem in [(3,QU),(4,ME),(5,HA),(6,NO)]:
    tr=pd.concat([panels[s] for s in ORDER[:k]],ignore_index=True)
    tf,yf=stream(stem); nps=int(yf.sum())
    for mname,mlab in (("gradient_boosting","GB"),("random_forest","RF"),("logistic_regression","LR"),
                       ("extra_trees","ET"),("mlp","MLP"),("isolation_forest","IF"),("learned_graph","GNN")):
        try: s=scores_for(mname,tr,tf.copy())
        except Exception: continue
        rk=online_rank(s)
        for lvl,tag in ((0.99,"rank1pct"),(0.999,"rank01pct")):
            a=rk>=lvl; tp=int((a&(yf==1)).sum()); fp=int((a&(yf==0)).sum())
            grid.append(dict(fold=NAMES[stem],model=mlab,method=tag,TP=tp,FP=fp,npos=nps))
    print(f"  {NAMES[stem]} done",flush=True)
json.dump(grid,open(GEN+"/live_detection/rank_grid_all_models.json","w"),indent=1)

# ---- (3) GNN negative control ----
print("=== GNN NEGATIVE CONTROL ===",flush=True)
PN=GEN+"/data/observed_panels"
neg=pd.concat([pd.read_csv(f"{PN}/socket_2024_quiet_control_{sp}.csv",dtype=str,keep_default_na=False) for sp in("train","test")],ignore_index=True)
neg["t"]=[t_obs(r) for _,r in neg.iterrows()]
neg=neg[neg["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
tr=pd.concat([panels[s] for s in ORDER],ignore_index=True)
try:
    s=scores_for("learned_graph",tr,neg.copy())
    rk=online_rank(s)
    npdt=None
    # simple NPDT reuse: import from stream_chrono would rerun grid; re-implement minimal
    def npdt_alerts(s,wfrac=0.3):
        n=len(s); w=max(200,int(wfrac*n)); step=max(50,w//2); alerts=set()
        for lo in range(0,n,step):
            win=s[lo:lo+w]
            if len(win)<50: break
            mu,sd=float(win.mean()),float(win.std())
            if sd==0: continue
            best=None;beste=None
            for z in (2,2.5,3,4,5,7,10):
                eps=mu+z*sd; above=win>eps; na=int(above.sum())
                if na==0: continue
                dmu=mu-float(win[~above].mean()); dsd=sd-float(win[~above].std())
                segs=int(np.diff(np.where(above)[0]).clip(min=0).astype(bool).size) or 1
                sc=(dmu/mu if mu else 0)+(dsd/sd if sd else 0); sc/=(na+segs**2)
                if best is None or sc>best: best,beste=sc,eps
            if beste is not None:
                for j in np.where(win>beste)[0]: alerts.add(lo+j)
        return sorted(alerts)
    ev=spot_alerts(s) or []
    print(f"  GNN quiet: max={s.max():.3f} EVT {len(ev)} | NPDT {len(npdt_alerts(s))} | rank0.1% {int((rk>=0.999).sum())}",flush=True)
    json.dump(dict(evt=len(ev),npdt=len(npdt_alerts(s)),rank=int((rk>=0.999).sum()),smax=float(s.max())),
              open(GEN+"/live_detection/gnn_negctl.json","w"))
except Exception as e:
    print("  GNN negctl FAILED:",type(e).__name__,e,flush=True)
print("JOB B DONE",flush=True)
