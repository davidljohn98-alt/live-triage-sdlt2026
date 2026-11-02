#!/usr/bin/env python3
"""Review Q3: ONE pre-specified policy = online rank threshold, ALL 4 evaluated
folds x ALL 7 models. No import of stream_chrono (avoids module-level grid rerun)."""
import sys, os, bisect
GEN="<ARTEFACT_ROOT>"
sys.path.insert(0,GEN); sys.path.insert(0,GEN+"/live_detection")
import numpy as np, pandas as pd
import train_models_per_case as TM
import train_score_graph_models as TG
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
panels={s:full(s) for s in ORDER}
def online_rank(s):
    out=np.zeros(len(s)); seen=[]
    for i,x in enumerate(s):
        out[i]=bisect.bisect_left(seen,x)/len(seen) if seen else 0.0
        bisect.insort(seen,float(x))
    return out
MODELS=[("gradient_boosting","GB"),("random_forest","RF"),("logistic_regression","LR"),
        ("extra_trees","ET"),("mlp","MLP"),("isolation_forest","IF"),("learned_graph","GNN")]
print(f"{'fold':8s}{'model':5s}  top1%: TP/FP prec recall | top0.1%: TP/FP prec recall")
for k,stem in [(3,"qubit_2022_bad_business_logic"),(4,"meter_io_2022_bad_business_logic"),
               (5,"harmony_horizon_2022_private_key_leak"),(6,"nomad_2022_authentication_bypass")]:
    tr=pd.concat([panels[s] for s in ORDER[:k]],ignore_index=True)
    te=panels[stem].copy()
    te["t"]=[t_obs(r) for _,r in te.iterrows()]
    te=te[te["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
    y=(te[TM.LABEL_COL].astype(str)=="1").astype(int).to_numpy(); npos=int(y.sum())
    for mname,mlab in MODELS:
        try:
            if mname=="learned_graph":
                sm=TG.learned_graph_scores(tr,te.copy())
                s=np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])
            else:
                out=TM.fit_score(mname,tr,te.copy())
                sm={str(r["flow_id"]):float(r["model_score"]) for _,r in out.iterrows()}
                s=np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])
        except Exception as e:
            print(f"{NAMES[stem]:8s}{mlab:5s}  ERROR {type(e).__name__}"); continue
        rk=online_rank(s)
        row=[]
        for lvl in (0.99,0.999):
            a=rk>=lvl; tp=int((a&(y==1)).sum()); fp=int((a&(y==0)).sum())
            prec=tp/(tp+fp) if tp+fp else 0.0
            row.append(f"{tp}/{fp} p={prec:.2f} r={100*tp/npos:.0f}%")
        print(f"{NAMES[stem]:8s}{mlab:5s}  {row[0]:24s} | {row[1]}")
