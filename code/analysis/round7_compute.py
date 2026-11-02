#!/usr/bin/env python3
"""Round-7: (1) exact pre-alert arithmetic; (2) GNN negctl full row; (3) value-splitting
evasion replay on Nomad; (4) committed-policy negctl burden check."""
import sys, math, bisect, json
GEN="<ARTEFACT_ROOT>"
sys.path.insert(0,GEN)
import numpy as np, pandas as pd
import train_models_per_case as TM
import train_score_graph_models as TG
P=GEN+"/data/observed_panels_v2"; PN=GEN+"/data/observed_panels"
CS,PO,MC,QU,ME,HA,NO=("chainswap_2021_authentication_bypass","poly_network_2021_authentication_bypass",
 "multichain_2022_authentication_bypass","qubit_2022_bad_business_logic",
 "meter_io_2022_bad_business_logic","harmony_horizon_2022_private_key_leak",
 "nomad_2022_authentication_bypass")
def full(s,base=None):
    b=base or P
    return pd.concat([pd.read_csv(f"{b}/{s}_{sp}.csv",dtype=str,keep_default_na=False) for sp in("train","test")],ignore_index=True)
def t_obs(r):
    ts=[]
    for c in("src_block_timestamp","dst_block_timestamp"):
        v=str(r.get(c,"")).strip()
        if v:
            try: ts.append(int(float(v)))
            except ValueError: pass
    return max(ts) if ts else None
def stream_df(df):
    te=df.copy(); te["t"]=[t_obs(r) for _,r in te.iterrows()]
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
def online_rank(s):
    out=np.zeros(len(s)); seen=[]
    for i,x in enumerate(s):
        out[i]=bisect.bisect_left(seen,x)/len(seen) if seen else 0.0
        bisect.insort(seen,float(x))
    return out
def score(mname,tr,te):
    if mname=="learned_graph":
        sm=TG.learned_graph_scores(tr,te.copy())
    else:
        out=TM.fit_score(mname,tr,te.copy())
        sm={str(r["flow_id"]):float(r["model_score"]) for _,r in out.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])

trN=pd.concat([full(s) for s in (CS,PO,MC,QU,ME,HA)],ignore_index=True)
teN,yN=stream_df(full(NO)); tN=teN["t"].astype(int).to_numpy(); npos=int(yN.sum())

print("=== (1) EXACT pre-alert arithmetic, Nomad ET EVT q=1e-3 ===",flush=True)
sE=score("extra_trees",trN,teN)
al=spot(sE); la=[i for i in al if yN[i]==1]; t1=tN[min(la)]
strictly_before=int(((tN<t1)&(yN==1)).sum()); at_or_before=int(((tN<=t1)&(yN==1)).sum())
still=npos-at_or_before
print(f"  linked strictly before first alert: {strictly_before}; at-or-before (incl. alerted): {at_or_before}; still to come: {still}/{npos} = {100*still/npos:.1f}%",flush=True)

print("=== (2) GNN negative-control full row ===",flush=True)
neg,_=stream_df(full("socket_2024_quiet_control",PN))
tr7=pd.concat([full(s) for s in (CS,PO,MC,QU,ME,HA,NO)],ignore_index=True)
sG=score("learned_graph",tr7,neg)
rk=online_rank(sG)
print(f"  GNN quiet: max={sG.max():.3f}  tau0.5={int((sG>=0.5).sum())}  EVT={len(spot(sG) or [])}  rank0.1%={int((rk>=0.999).sum())}",flush=True)

print("=== (3) VALUE-SPLITTING evasion replay, Nomad (each linked flow -> 3 thirds) ===",flush=True)
pert=[]
for _,r in teN.iterrows():
    if str(r[TM.LABEL_COL])=="1":
        for k in range(3):
            rr=r.copy()
            for c in ("src_amount","dst_amount","src_amount_raw","dst_amount_raw"):
                v=str(rr.get(c,"")).strip()
                if v:
                    try: rr[c]=str(float(v)/3.0)
                    except ValueError: pass
            rr["flow_id"]=str(r["flow_id"])+f"_s{k}"
            pert.append(rr)
    else:
        pert.append(r)
teP=pd.DataFrame(pert).reset_index(drop=True)
teP,yP=stream_df(teP); nposP=int(yP.sum())
print(f"  perturbed stream: {len(teP)} flows, {nposP} linked",flush=True)
for m,ml in (("extra_trees","ET"),("gradient_boosting","GB")):
    sP=score(m,trN,teP)
    alP=spot(sP) or []
    tp=sum(1 for i in alP if yP[i]==1); fp=len(alP)-tp
    rkP=online_rank(sP); a=rkP>=0.999
    tpr=int((a&(yP==1)).sum()); fpr=int((a&(yP==0)).sum())
    print(f"  {ml}: EVT {tp}/{fp} (prec {tp/len(alP) if alP else 0:.2f}; was 13/2 resp 10/1) | rank0.1% {tpr}/{fpr} (was 9/7 resp 7/3)",flush=True)
print("DONE",flush=True)
