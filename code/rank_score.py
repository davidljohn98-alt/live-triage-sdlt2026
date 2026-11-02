#!/usr/bin/env python3
"""A better risk-score metric for live detection.

Raw model probabilities do not transfer between incidents (Sec. score scales).
This evaluates an ONLINE RELATIVE score instead: each arriving flow is scored by
its percentile among the flows seen so far in the same stream, so the quantity
thresholded is dimensionless and comparable across incidents by construction.
No look-ahead: the percentile uses only already-arrived flows.

Compares, per fold and model:
  raw   - fixed tau on the model probability          (current baseline)
  rank  - fixed tau on the online percentile          (proposed metric)
Writes rank_score_results.json / .csv
"""
import sys, os, json, csv
from datetime import datetime, timezone
HERE=os.path.dirname(os.path.abspath(__file__)); GEN=os.path.dirname(HERE)
sys.path.insert(0,GEN); sys.path.insert(0,HERE)
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from stream_chrono import panels, ORDER, scores_for, t_obs, eval_alerts
import train_models_per_case as TM
LBL=TM.LABEL_COL
stems=[s for s,_ in ORDER]; names={s:l for s,l in ORDER}
MODELS=[("gradient_boosting","GB"),("random_forest","RF"),("extra_trees","ET"),("logistic_regression","LR")]

def online_rank(s):
    """percentile of each element among all strictly-earlier elements (no look-ahead)"""
    n=len(s); out=np.zeros(n)
    # order statistics maintained incrementally via sorted insertion
    import bisect
    seen=[]
    for i,x in enumerate(s):
        out[i]= (bisect.bisect_left(seen,x)/len(seen)) if seen else 0.0
        bisect.insort(seen,x)
    return out

rows=[]
for k in range(2,len(stems)):
    stem=stems[k]; lib=stems[:k]
    tr=pd.concat([panels[x] for x in lib],ignore_index=True)
    te=panels[stem].copy()
    te["t"]=[t_obs(r) for _,r in te.iterrows()]
    te=te[te["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
    y=(te[LBL].astype(str)=="1").astype(int).to_numpy()
    day=np.array([datetime.fromtimestamp(int(x),tz=timezone.utc).date().isoformat() for x in te["t"].astype(int)])
    if y.sum()==0: continue
    for mname,mlab in MODELS:
        try: s=scores_for(mname,tr,te.copy())
        except Exception as e:
            print(f"  {names[stem]} {mlab}: ERROR {str(e)[:60]}"); continue
        r=online_rank(s)
        for tau in (0.99,0.999):
            raw_thr=np.quantile(s,tau)   # what a fixed prob cutoff cannot know
            ev_rank=eval_alerts(np.where(r>=tau)[0],y,day)
            rows.append(dict(fold=names[stem],library=k,model=mlab,metric="rank",tau=tau,
                             TP=ev_rank["TP"],FP=ev_rank["FP"],precision=round(ev_rank["precision"],3),
                             lag=ev_rank["detection_lag_days"]))
        for tau in (0.5,0.9):
            ev_raw=eval_alerts(np.where(s>=tau)[0],y,day)
            rows.append(dict(fold=names[stem],library=k,model=mlab,metric="raw",tau=tau,
                             TP=ev_raw["TP"],FP=ev_raw["FP"],precision=round(ev_raw["precision"],3),
                             lag=ev_raw["detection_lag_days"]))
        print(f"  {names[stem]:16s} {mlab:3s} rank@0.999 {rows[-3]['TP']}TP/{rows[-3]['FP']}FP | raw@0.5 {rows[-2]['TP']}TP/{rows[-2]['FP']}FP")
json.dump(rows,open(os.path.join(HERE,"rank_score_results.json"),"w"),indent=1)
with open(os.path.join(HERE,"rank_score_results.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote rank_score_results.json/.csv")
