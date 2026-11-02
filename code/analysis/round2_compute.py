#!/usr/bin/env python3
"""Round-2 review computations: ordering spreads, permuted negctl, out-of-spec EVT,
hour lags, pre-alert arrival fraction."""
import sys, math, bisect, json
GEN="<ARTEFACT_ROOT>"
sys.path.insert(0,GEN)
import numpy as np, pandas as pd
import train_models_per_case as TM
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
def stream(stem,base=None):
    te=full(stem,base); te["t"]=[t_obs(r) for _,r in te.iterrows()]
    te=te[te["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
    y=(te[TM.LABEL_COL].astype(str)=="1").astype(int).to_numpy()
    return te,y
def spot(s,q=1e-3,init_q=0.98,min_n=500):
    n=len(s); ni=max(150,int(0.15*n))
    if n<min_n:
        if ni>=n: return "calibration exceeds stream"
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
    out=TM.fit_score(mname,tr,te.copy())
    sm={str(r["flow_id"]):float(r["model_score"]) for _,r in out.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])
def online_rank(s):
    out=np.zeros(len(s)); seen=[]
    for i,x in enumerate(s):
        out[i]=bisect.bisect_left(seen,x)/len(seen) if seen else 0.0
        bisect.insort(seen,float(x))
    return out
def npdt(s,wfrac=0.3):
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

ORDERINGS={"chrono":[CS,PO,MC,QU,ME,HA],"meter-first":[ME,CS,PO,MC,QU,HA],
           "meter-second":[CS,ME,PO,MC,QU,HA],"meter-last":[CS,PO,MC,QU,HA,ME],
           "reversed":[HA,ME,QU,MC,PO,CS]}
teN,yN=stream(NO); tN=teN["t"].astype(int).to_numpy(); nposN=int(yN.sum())
first_linked=tN[np.argmax(yN==1)]

print("=== (A) ORDERING SPREAD, full six-exploit library, EVT q=1e-3 on Nomad ===",flush=True)
for m,ml in (("extra_trees","ET"),("gradient_boosting","GB")):
    res=[]
    for on,order in ORDERINGS.items():
        tr=pd.concat([full(s) for s in order],ignore_index=True)
        s=score(m,tr,teN); al=spot(s)
        tp=sum(1 for i in al if yN[i]==1); fp=len(al)-tp
        res.append((on,tp,fp,round(tp/len(al),2) if al else 0))
    print(f"  {ml}: "+" | ".join(f"{o}: {tp}/{fp} p={p}" for o,tp,fp,p in res),flush=True)

print("=== (B) NEGATIVE CONTROL, GB, permuted 7-library orderings ===",flush=True)
neg,_=stream("socket_2024_quiet_control",PN)
LIB7={"chrono":[CS,PO,MC,QU,ME,HA,NO],"rev":[NO,HA,ME,QU,MC,PO,CS],
      "meter-first":[ME,CS,PO,MC,QU,HA,NO],"nomad-first":[NO,CS,PO,MC,QU,ME,HA]}
for on,order in LIB7.items():
    tr=pd.concat([full(s) for s in order],ignore_index=True)
    s=score("gradient_boosting",tr,neg)
    ev=spot(s); rk=online_rank(s)
    print(f"  {on:12s}: EVT {len(ev) if isinstance(ev,list) else ev} | NPDT {len(npdt(s))} | rank0.1% {int((rk>=0.999).sum())} | max {s.max():.3f}",flush=True)

print("=== (C) OUT-OF-SPEC EVT DIAGNOSTIC, Harmony (275 flows) + Qubit check ===",flush=True)
teH,yH=stream(HA); trH=pd.concat([full(s) for s in (CS,PO,MC,QU,ME)],ignore_index=True)
print("  Qubit: calibration max(150,0.15*76)=150 > 76-flow stream ->", spot(np.zeros(76),min_n=0))
for m,ml in (("gradient_boosting","GB"),("random_forest","RF"),("extra_trees","ET"),
             ("logistic_regression","LR"),("mlp","MLP"),("isolation_forest","IF")):
    s=score(m,trH,teH); al=spot(s,min_n=0)
    if isinstance(al,str): print(f"  Harmony {ml}: {al}",flush=True); continue
    tp=sum(1 for i in al if yH[i]==1); fp=len(al)-tp
    print(f"  Harmony {ml}: TP{tp}/FP{fp} ({len(al)} alerts, out-of-spec: 3 calibration peaks < 10 required)",flush=True)

print("=== (D) HOUR LAGS for oracle rows: Qubit LR tau0.5, Meter GNN tau0.9 ===",flush=True)
teQ,yQ=stream(QU); tQ=teQ["t"].astype(int).to_numpy()
trQ=pd.concat([full(s) for s in (CS,PO,MC)],ignore_index=True)
sQ=score("logistic_regression",trQ,teQ)
alQ=np.where(sQ>=0.5)[0]; la=[i for i in alQ if yQ[i]==1]
print(f"  Qubit LR tau0.5: lag {(tQ[min(la)]-tQ[np.argmax(yQ==1)])/3600:.1f} h",flush=True)
import train_score_graph_models as TG
teM,yM=stream(ME); tM=teM["t"].astype(int).to_numpy()
trM=pd.concat([full(s) for s in (CS,PO,MC,QU)],ignore_index=True)
sm=TG.learned_graph_scores(trM,teM.copy())
sM=np.array([sm.get(str(f),0.0) for f in teM["flow_id"].astype(str)])
alM=np.where(sM>=0.9)[0]; laM=[i for i in alM if yM[i]==1]
print(f"  Meter GNN tau0.9: lag {(tM[min(laM)]-tM[np.argmax(yM==1)])/3600:.1f} h" if laM else "  Meter GNN: no linked alert?!",flush=True)

print("=== (E) FRACTION OF NOMAD'S 615 ARRIVED BEFORE FIRST ALERT (ET, GB EVT q=1e-3, chrono) ===",flush=True)
trN=pd.concat([full(s) for s in (CS,PO,MC,QU,ME,HA)],ignore_index=True)
for m,ml in (("extra_trees","ET"),("gradient_boosting","GB")):
    s=score(m,trN,teN); al=spot(s)
    la=[i for i in al if yN[i]==1]
    t_first=tN[min(la)]
    arrived=int(((tN<t_first)&(yN==1)).sum())
    print(f"  {ml}: first linked alert at +{(t_first-first_linked)/3600:.1f} h; {arrived}/{nposN} linked flows ({100*arrived/nposN:.0f}%) had already arrived",flush=True)
print("DONE",flush=True)
