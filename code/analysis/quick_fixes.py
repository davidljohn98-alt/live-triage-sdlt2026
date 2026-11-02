#!/usr/bin/env python3
"""JobA: (1) trivial heuristic baselines, (2) hour-granularity lag on Nomad,
(3) Table-4 recompute excluding the Multichain diagnostic."""
import sys, os, bisect, json
GEN="<ARTEFACT_ROOT>"
sys.path.insert(0,GEN); sys.path.insert(0,GEN+"/live_detection")
import numpy as np, pandas as pd
import train_models_per_case as TM
P=GEN+"/data/observed_panels_v2"
ORDER=["chainswap_2021_authentication_bypass","poly_network_2021_authentication_bypass",
 "multichain_2022_authentication_bypass","qubit_2022_bad_business_logic",
 "meter_io_2022_bad_business_logic","harmony_horizon_2022_private_key_leak",
 "nomad_2022_authentication_bypass"]
NAMES=dict(zip(ORDER,["ChainSwap","Poly","Multichain","Qubit","Meter","Harmony","Nomad"]))
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
def stream(stem):
    te=full(stem); te["t"]=[t_obs(r) for _,r in te.iterrows()]
    te=te[te["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
    y=(te[TM.LABEL_COL].astype(str)=="1").astype(int).to_numpy()
    return te,y
def online_rank(s):
    out=np.zeros(len(s)); seen=[]
    for i,x in enumerate(s):
        out[i]=bisect.bisect_left(seen,x)/len(seen) if seen else 0.0
        bisect.insort(seen,float(x))
    return out
FOLDS=ORDER[3:]

print("=== (1) HEURISTIC BASELINES (no learning) ===")
res={}
for stem in FOLDS:
    te,y=stream(stem); npos=int(y.sum()); nm=NAMES[stem]
    amt=np.zeros(len(te))
    for i,(_,r) in enumerate(te.iterrows()):
        v=0.0
        for c in ("src_amount_usd","dst_amount_usd","src_amount","dst_amount"):
            x=str(r.get(c,"")).strip()
            if x:
                try: v=max(v,abs(float(x)))
                except ValueError: pass
        amt[i]=np.log1p(v)
    rk=online_rank(amt)
    out={}
    for lvl,tag in ((0.999,"amount rank 0.1%"),(0.99,"amount rank 1%")):
        a=rk>=lvl; tp=int((a&(y==1)).sum()); fp=int((a&(y==0)).sum())
        out[tag]=dict(TP=tp,FP=fp,prec=round(tp/(tp+fp),2) if tp+fp else 0.0,recall=round(tp/npos,3))
    # route novelty after 20% warmup
    warm=int(0.2*len(te)); seen=set(); alerts=[]
    cols=[c for c in ("bridge","src_chain","dst_chain","route") if c in te.columns]
    for i,(_,r) in enumerate(te.iterrows()):
        key=tuple(str(r.get(c,"")) for c in cols)
        if key not in seen:
            if i>=warm: alerts.append(i)
            seen.add(key)
    a=np.zeros(len(te),bool); a[alerts]=True
    tp=int((a&(y==1)).sum()); fp=int((a&(y==0)).sum())
    out["route novelty"]=dict(TP=tp,FP=fp,prec=round(tp/(tp+fp),2) if tp+fp else 0.0,recall=round(tp/npos,3))
    res[nm]=out
    print(f"  {nm:8s} npos={npos:3d} :: "+" | ".join(f"{k}: TP{v['TP']}/FP{v['FP']} p={v['prec']} r={v['recall']}" for k,v in out.items()))
json.dump(res,open(f"{GEN}/live_detection/heuristic_baselines.json","w"),indent=1)

print("\n=== (2) HOUR-GRANULARITY LAG, Nomad, EVT q=1e-3, ET & GB ===")
import math
def spot_alerts(s,q,n_init=None,init_q=0.98):
    n=len(s); n_init=n_init or max(150,int(0.15*n))
    if n<500: return None
    init=s[:n_init]; t=float(np.quantile(init,init_q))
    peaks=list(init[init>t]-t)
    def zq(Nt,ncur):
        if len(peaks)<10: return None
        pk=np.array(peaks); m,v=pk.mean(),pk.var()
        if v<=0: return t+m
        gamma=0.5*(1-m*m/v); sigma=0.5*m*(m*m/v+1)
        r=q*ncur/max(Nt,1)
        try:
            if abs(gamma)<1e-6: return t - sigma*math.log(r)
            return t + (sigma/gamma)*((r**(-gamma))-1)
        except (OverflowError, ValueError): return None
    z=zq(len(peaks),n_init) or float(np.quantile(init,0.999))
    alerts=[]; Nt=len(peaks); ncur=n_init
    for i in range(n_init,n):
        x=s[i]; ncur+=1
        if x>z: alerts.append(i)
        elif x>t:
            peaks.append(x-t); Nt+=1
            z2=zq(Nt,ncur)
            if z2 is not None: z=z2
    return alerts

nomad="nomad_2022_authentication_bypass"
te,y=stream(nomad); t=te["t"].astype(int).to_numpy()
tr=pd.concat([full(s) for s in ORDER[:6]],ignore_index=True)
first_linked=t[np.argmax(y==1)]
for mname,mlab in (("extra_trees","ET"),("gradient_boosting","GB")):
    out=TM.fit_score(mname,tr,te.copy())
    sm={str(r["flow_id"]):float(r["model_score"]) for _,r in out.iterrows()}
    sc=np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])
    al=spot_alerts(sc,1e-3) or []
    la=[i for i in al if y[i]==1]
    if la:
        first_alert=t[min(la)]
        lag_h=(first_alert-first_linked)/3600.0
        print(f"  {mlab}: first incident-linked EVT alert {lag_h:.1f} h after first incident-linked arrival "
              f"({len(la)} linked / {len(al)} alerts)")
    else:
        print(f"  {mlab}: no linked alert")

print("\n=== (3) TABLE 4 RECOMPUTED — evaluated folds only (no Multichain diagnostic) ===")
rr=json.load(open(GEN+"/live_detection/rank_score_results.json"))
ev=[x for x in rr if x.get("fold")!="Multichain"]
import collections
def agg(metric,tau):
    rows=[x for x in ev if x["metric"]==metric and abs(x["tau"]-tau)<1e-9]
    n=len(rows)
    silent=sum(1 for x in rows if (x["TP"]+x.get("FP",0))==0)
    flood=sum(1 for x in rows if (x["TP"]+x.get("FP",0))>=200)
    amin=min(x["TP"]+x.get("FP",0) for x in rows); amax=max(x["TP"]+x.get("FP",0) for x in rows)
    rec=sum(1 for x in rows if x["TP"]>0)
    return n,silent,flood,amin,amax,rec
for metric,tau,lab in (("raw",0.5,"Probability, tau=0.5"),("raw",0.9,"Probability, tau=0.9"),
                        ("rank",0.99,"Rank, top 1%"),("rank",0.999,"Rank, top 0.1%")):
    try:
        n,si,fl,a0,a1,rc=agg(metric,tau)
        print(f"  {lab:22s}: n={n}  silent {si}/{n}  flooding {fl}  alerts {a0}-{a1}  recovers {rc}/{n}")
    except Exception as e:
        print(f"  {lab}: ERROR {e} — keys: {sorted(set(x['metric'] for x in ev))}, taus {sorted(set(x['tau'] for x in ev))}")
