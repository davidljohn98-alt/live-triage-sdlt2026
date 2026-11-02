#!/usr/bin/env python3
"""Per-alert attribution: why did this flow ping? (the supervisor's priority, 20 Aug 2026)

For every alert raised by the committed adaptive policy on the fullest-library
fold, computes Shapley values over the fitted scorer and folds them into four
analyst-readable factor groups:
    route     - bridge / source chain / destination chain / route identity
    amount    - transferred amounts and their ratio
    timing    - bridge lag
    address   - sender and recipient history in the training library
Writes alert_attribution.json / .csv and prints a reason line per alert.
"""
import sys, os, json, csv
from datetime import datetime, timezone
HERE=os.path.dirname(os.path.abspath(__file__)); GEN=os.path.dirname(HERE)
sys.path.insert(0,GEN); sys.path.insert(0,HERE)
import numpy as np, pandas as pd, shap
from stream_chrono import panels, ORDER, t_obs, spot_alerts
import train_models_per_case as TM

LBL=TM.LABEL_COL
stems=[s for s,_ in ORDER]; names={s:l for s,l in ORDER}
TEST=stems[-1]; LIB=stems[:-1]
MODEL="extra_trees"          # best performer at full library (learning curve)

GROUP={}
for c in TM.CATEGORICAL_COLS: GROUP[c]="route"
GROUP.update({"log_src_amount":"amount","log_dst_amount":"amount","amount_ratio":"amount",
              "log_lag_seconds":"timing","sender_recipient_equal":"address",
              "src_sender_freq":"address","dst_recipient_freq":"address"})

tr=pd.concat([panels[s] for s in LIB],ignore_index=True)
te=panels[TEST].copy()
te["t"]=[t_obs(r) for _,r in te.iterrows()]
te=te[te["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
y=(te[LBL].astype(str)=="1").astype(int).to_numpy()

# rebuild the exact fitted pipeline
fm=TM.build_frequency_maps(tr)
Xtr=TM.featurise(tr,*fm); Xte=TM.featurise(te,*fm)
ytr=(tr[LBL].astype(str)=="1").astype(int).to_numpy()
pre=TM._build_preprocessor()
est,_=TM.make_model(MODEL)
Atr=pre.fit_transform(Xtr); Ate=pre.transform(Xte)
Atr=Atr.toarray() if hasattr(Atr,"toarray") else Atr
Ate=Ate.toarray() if hasattr(Ate,"toarray") else Ate
est.fit(Atr,ytr)
scores=est.predict_proba(Ate)[:,1]
print(f"fold {names[TEST]}: {len(te):,} flows, {int(y.sum())} linked, scorer {MODEL}")

alerts=spot_alerts(scores,1e-3) or []
print(f"committed adaptive policy raised {len(alerts)} alerts")
if not alerts: raise SystemExit

feat_names=list(pre.get_feature_names_out())
expl=shap.TreeExplainer(est)
sv=expl.shap_values(Ate[alerts])
if isinstance(sv,list): sv=sv[1]
if sv.ndim==3: sv=sv[:,:,1]

def group_of(fn):
    base=fn.split("__",1)[1] if "__" in fn else fn
    for k,g in GROUP.items():
        if base==k or base.startswith(k+"_"): return g
    return "other"

rows=[]
for j,i in enumerate(alerts):
    contrib={}
    for fn,v in zip(feat_names,sv[j]):
        contrib[group_of(fn)]=contrib.get(group_of(fn),0.0)+float(v)
    tot=sum(abs(v) for v in contrib.values()) or 1.0
    ranked=sorted(contrib.items(),key=lambda x:-abs(x[1]))
    reason=", ".join(f"{g} {100*abs(v)/tot:.0f}%" for g,v in ranked[:3])
    ts=datetime.fromtimestamp(int(te.loc[i,"t"]),tz=timezone.utc)
    rows.append(dict(flow_index=int(i), utc=ts.strftime("%Y-%m-%d %H:%M"),
                     score=round(float(scores[i]),4), incident_linked=int(y[i]),
                     top_factors=reason,
                     **{f"shap_{g}":round(contrib.get(g,0.0),5) for g in ("route","amount","timing","address")}))
    print(f"  alert {j+1:2d} {ts:%d %b %H:%M}  score {scores[i]:.3f}  linked={'YES' if y[i] else 'no ':3s}  because: {reason}")

json.dump(rows,open(os.path.join(HERE,"alert_attribution.json"),"w"),indent=1)
with open(os.path.join(HERE,"alert_attribution.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote alert_attribution.json/.csv")
