#!/usr/bin/env python3
"""Q6: does the row-order variance persist under reseeding? 10 seeds, full 6-library,
Nomad, EVT q=1e-3, ET+GB, chronological ordering."""
import sys, math, json
GEN="<ARTEFACT_ROOT>"
sys.path.insert(0,GEN)
import numpy as np, pandas as pd
import train_models_per_case as TM
P=GEN+"/data/observed_panels_v2"
CASES=("chainswap_2021_authentication_bypass","poly_network_2021_authentication_bypass",
 "multichain_2022_authentication_bypass","qubit_2022_bad_business_logic",
 "meter_io_2022_bad_business_logic","harmony_horizon_2022_private_key_leak")
NO="nomad_2022_authentication_bypass"
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
def spot(s,q=1e-3,init_q=0.98):
    n=len(s); ni=max(150,int(0.15*n)); init=s[:ni]; t=float(np.quantile(init,init_q)); peaks=list(init[init>t]-t)
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
tr=pd.concat([full(s) for s in CASES],ignore_index=True)
te=full(NO); te["t"]=[t_obs(r) for _,r in te.iterrows()]
te=te[te["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
y=(te[TM.LABEL_COL].astype(str)=="1").astype(int).to_numpy()
res={}
for m,ml in (("extra_trees","ET"),("gradient_boosting","GB")):
    tps=[]
    for seed in range(42,52):
        TM.RANDOM_STATE=seed
        out=TM.fit_score(m,tr,te.copy())
        sm={str(r["flow_id"]):float(r["model_score"]) for _,r in out.iterrows()}
        s=np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])
        al=spot(s) or []
        tp=sum(1 for i in al if y[i]==1); fp=len(al)-tp
        tps.append((seed,tp,fp))
        print(f"{ml} seed {seed}: {tp}/{fp}",flush=True)
    res[ml]=tps
json.dump(res,open(GEN+"/live_detection/multiseed_nomad.json","w"),indent=1)
tps_et=[t for _,t,_ in res["ET"]]; tps_gb=[t for _,t,_ in res["GB"]]
print(f"SUMMARY: ET recoveries over 10 seeds {min(tps_et)}-{max(tps_et)}; GB {min(tps_gb)}-{max(tps_gb)}",flush=True)
