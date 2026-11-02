#!/usr/bin/env python3
"""No-incident negative control.

Replays a 14-day stream of real cross-chain traffic that contains NO
incident-linked flow, scored by the full incident library and read through the
same committed alert interfaces used on the detection folds. Everything the
policy raises here is by construction a false alert, so this measures the
quiet-period alert burden that incident-centred collection cannot show.
Writes negative_control.json.
"""
import sys, os, json
from datetime import datetime, timezone
HERE=os.path.dirname(os.path.abspath(__file__)); GEN=os.path.dirname(HERE)
sys.path.insert(0,GEN); sys.path.insert(0,HERE)
import numpy as np, pandas as pd
from stream_chrono import panels, ORDER, scores_for, t_obs, spot_alerts, npdt_alerts, fixed_tau
import train_models_per_case as TM

CASE="socket_2024_quiet_control"
P=os.path.join(GEN,"data/observed_panels")
def load(n): return pd.read_csv(f"{P}/{n}.csv",dtype=str,keep_default_na=False)
neg=pd.concat([load(f"{CASE}_train"),load(f"{CASE}_test")],ignore_index=True)
neg["t"]=[t_obs(r) for _,r in neg.iterrows()]
neg=neg[neg["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
day=np.array([datetime.fromtimestamp(int(x),tz=timezone.utc).date().isoformat() for x in neg["t"].astype(int)])
ndays=len(set(day))
print(f"negative-control stream: {len(neg):,} flows over {ndays} days, "
      f"{int((neg[TM.LABEL_COL].astype(str)=='1').sum())} incident-linked (expected 0)")

tr=pd.concat([panels[s] for s,_ in ORDER],ignore_index=True)
print(f"scored by the full library: {len(ORDER)} incidents, {len(tr):,} training flows")

out={"n_flows":len(neg),"n_days":ndays,"library_incidents":len(ORDER),"models":{}}
for mname,mlab in (("extra_trees","ET"),("gradient_boosting","GB"),("random_forest","RF"),
                   ("logistic_regression","LR"),("mlp","MLP"),("isolation_forest","IF")):
    try: s=scores_for(mname,tr,neg.copy())
    except Exception as e:
        print(f"  {mlab}: ERROR {type(e).__name__}"); continue
    import bisect as _b
    _seen=[]; rank=[]
    for _x in s:
        rank.append(_b.bisect_left(_seen,_x)/len(_seen) if _seen else 0.0); _b.insort(_seen,float(_x))
    rank=__import__("numpy").array(rank)  # ONLINE (causal) rank, matching the paper definition
    res={}
    for name,idx in (("tau0.5",fixed_tau(s,0.5)),("tau0.9",fixed_tau(s,0.9)),
                     ("EVT q=1e-3",spot_alerts(s,1e-3) or []),
                     ("NPDT",npdt_alerts(s)),
                     ("rank top0.1%",np.where(rank>=0.999)[0])):
        n=len(idx)
        res[name]=dict(false_alerts=int(n), per_day=round(n/ndays,2))
    out["models"][mlab]=dict(score_max=round(float(s.max()),4),score_p99=round(float(np.quantile(s,0.99)),5),thresholds=res)
    line=" | ".join(f"{k}: {v['false_alerts']} ({v['per_day']}/day)" for k,v in res.items())
    print(f"  {mlab:3s} max={s.max():.3f}  {line}")
json.dump(out,open(os.path.join(HERE,"negative_control.json"),"w"),indent=1)
print("wrote negative_control.json")
