#!/usr/bin/env python3
"""(A) Seeded address-propagation baseline on Nomad's post-onset regime.
(B) Matched-budget recall@10/@20 for 7 models + amount heuristic x 4 folds."""
import sys, math, bisect, json
GEN="<ARTEFACT_ROOT>"
sys.path.insert(0,GEN)
import numpy as np, pandas as pd
import train_models_per_case as TM
import train_score_graph_models as TG
P=GEN+"/data/observed_panels_v2"
CS,PO,MC,QU,ME,HA,NO=("chainswap_2021_authentication_bypass","poly_network_2021_authentication_bypass",
 "multichain_2022_authentication_bypass","qubit_2022_bad_business_logic",
 "meter_io_2022_bad_business_logic","harmony_horizon_2022_private_key_leak",
 "nomad_2022_authentication_bypass")
NAMES={QU:"Qubit",ME:"Meter",HA:"Harmony",NO:"Nomad"}
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
def spot(s,q=1e-3,init_q=0.98):
    n=len(s); ni=max(150,int(0.15*n))
    if n<500: return None
    init=s[:ni]; t=float(np.quantile(init,init_q)); peaks=list(init[init>t]-t)
    def zq(Nt,nc):
        if len(peaks)<10: return None
        pk=np.array(peaks); m,v=pk.mean(),pk.var()
        if v<=0: return t+m
        g=0.5*(1-m*m/v); sg=0.5*m*(m*m/v+1); r=q*nc/max(Nt,1)
        try:
            if abs(g)<1e-6: return t-sg*math.log(r)
            return t+(sg/g)*((r**(-g))-1)
        except (OverflowError,ValueError): return None
    z=zq(len(peaks),ni) or float(np.quantile(init,0.999))
    al=[]; Nt=len(peaks); nc=ni
    for i in range(ni,n):
        x=s[i]; nc+=1
        if x>z: al.append(i)
        elif x>t:
            peaks.append(x-t); Nt+=1
            z2=zq(Nt,nc)
            if z2 is not None: z=z2
    return al
def score(mname,tr,te):
    if mname=="learned_graph":
        sm=TG.learned_graph_scores(tr,te.copy())
    else:
        out=TM.fit_score(mname,tr,te.copy())
        sm={str(r["flow_id"]):float(r["model_score"]) for _,r in out.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])

ADDR=("src_sender","src_recipient","dst_sender","dst_recipient")
def addrs(row):
    return {str(row.get(c,"")).strip().lower() for c in ADDR if str(row.get(c,"")).strip()}

print("=== (A) SEEDED ADDRESS PROPAGATION, Nomad post-onset ===",flush=True)
trN=pd.concat([full(s) for s in (CS,PO,MC,QU,ME,HA)],ignore_index=True)
teN,yN=stream(NO); tN=teN["t"].astype(int).to_numpy(); npos=int(yN.sum())
sE=score("extra_trees",trN,teN)
al=spot(sE); la=[i for i in al if yN[i]==1]; i1=min(la); t1=tN[i1]
seed=addrs(teN.iloc[i1])
post=[i for i in range(len(teN)) if tN[i]>t1]
npos_post=sum(1 for i in post if yN[i]==1)
print(f"  seed = first linked EVT alert (index {i1}, {len(seed)} addresses); post-onset flows: {len(post)}, linked {npos_post}",flush=True)
# (i) propagating taint
S1=set(seed); tp=fp=0
for i in post:
    A=addrs(teN.iloc[i])
    if A & S1:
        if yN[i]==1: tp+=1
        else: fp+=1
        S1 |= A
print(f"  propagating taint:   TP {tp} / FP {fp}  recall {100*tp/npos_post:.1f}%  precision {tp/(tp+fp) if tp+fp else 0:.2f}  (address set grows to {len(S1)})",flush=True)
# (ii) seed-only matching
tp2=fp2=0
for i in post:
    if addrs(teN.iloc[i]) & seed:
        if yN[i]==1: tp2+=1
        else: fp2+=1
print(f"  seed-only matching:  TP {tp2} / FP {fp2}  recall {100*tp2/npos_post:.1f}%  precision {tp2/(tp2+fp2) if tp2+fp2 else 0:.2f}",flush=True)

print("=== (B) MATCHED-BUDGET recall@10/@20, all models + amount heuristic ===",flush=True)
LIBS={QU:(CS,PO,MC),ME:(CS,PO,MC,QU),HA:(CS,PO,MC,QU,ME),NO:(CS,PO,MC,QU,ME,HA)}
MODELS=[("gradient_boosting","GB"),("random_forest","RF"),("extra_trees","ET"),("logistic_regression","LR"),
        ("mlp","MLP"),("isolation_forest","IF"),("learned_graph","GNN")]
out={}
for stem in (QU,ME,HA,NO):
    tr=pd.concat([full(s) for s in LIBS[stem]],ignore_index=True)
    te,y=stream(stem); nps=int(y.sum()); out[NAMES[stem]]={}
    for m,ml in MODELS:
        s=score(m,tr,te)
        order=np.argsort(-s,kind="stable")
        for B in (10,20):
            out[NAMES[stem]].setdefault(ml,{})[B]=int(y[order[:B]].sum())
    amt=np.zeros(len(te))
    for i,(_,r) in enumerate(te.iterrows()):
        v=0.0
        for c in ("src_amount","dst_amount"):
            x=str(r.get(c,"")).strip()
            if x:
                try: v=max(v,abs(float(x)))
                except ValueError: pass
        amt[i]=np.log1p(v)
    order=np.argsort(-amt,kind="stable")
    for B in (10,20):
        out[NAMES[stem]].setdefault("Amount",{})[B]=int(y[order[:B]].sum())
    out[NAMES[stem]]["npos"]=nps
    print(f"  {NAMES[stem]} done (npos {nps}): "+", ".join(f"{k}@10={v[10]}" for k,v in out[NAMES[stem]].items() if k!='npos'),flush=True)
json.dump(out,open(GEN+"/live_detection/matched_budget.json","w"),indent=1)
print("DONE",flush=True)
