#!/usr/bin/env python3
"""Library-size learning curve (the library-size saturation question, 20 Aug 2026).

Holds the test incident fixed and grows the training library along the
chronological order, so the only thing varying is how many prior incidents the
scorer has seen. Reports average precision and the committed-alert outcome at
each library size, for every model.
Writes learning_curve.json + learning_curve.csv.
"""
import sys, os, json, csv
HERE = os.path.dirname(os.path.abspath(__file__)); GEN = os.path.dirname(HERE)
sys.path.insert(0, GEN); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from stream_chrono import panels, ORDER, scores_for, t_obs, spot_alerts, eval_alerts, fixed_tau
import train_models_per_case as TM
from datetime import datetime, timezone

LBL = TM.LABEL_COL
stems = [s for s, _ in ORDER]; names = {s: l for s, l in ORDER}
TEST = stems[-1]                      # last incident chronologically
LIB  = stems[:-1]                     # everything before it
MODELS = [("gradient_boosting","GB"), ("random_forest","RF"), ("logistic_regression","LR"),
          ("extra_trees","ET"), ("mlp","MLP"), ("isolation_forest","IF"), ("learned_graph","GNN")]

te = panels[TEST].copy()
te["t"] = [t_obs(r) for _, r in te.iterrows()]
te = te[te["t"].notna()].sort_values("t", kind="stable").reset_index(drop=True)
y = (te[LBL].astype(str) == "1").astype(int).to_numpy()
day = np.array([datetime.fromtimestamp(int(x), tz=timezone.utc).date().isoformat()
                for x in te["t"].astype(int)])
print(f"test incident: {names[TEST]}  ({len(te):,} flows, {int(y.sum())} linked)")

rows = []
for k in range(1, len(LIB) + 1):
    lib = LIB[:k]
    tr = pd.concat([panels[s] for s in lib], ignore_index=True)
    for mname, mlab in MODELS:
        try:
            s = scores_for(mname, tr, te.copy())
            ap = float(average_precision_score(y, s))
            spot = spot_alerts(s, 1e-3)
            ev = eval_alerts(spot, y, day) if spot else {"TP":0,"FP":0,"precision":0.0}
            evf = eval_alerts(fixed_tau(s, 0.5), y, day)
            p10 = float(y[np.argsort(-s, kind="stable")[:10]].mean())
            rows.append(dict(library_size=k, library=";".join(names[x] for x in lib),
                             model=mlab, ap=round(ap,4), p_at_10=round(p10,3),
                             evt_tp=ev["TP"], evt_fp=ev["FP"], evt_precision=round(ev["precision"],3),
                             fixed_tp=evf["TP"], fixed_fp=evf["FP"]))
            print(f"  lib={k} {mlab:4s} AP={ap:.3f} P@10={p10:.0%} EVT {ev['TP']}TP/{ev['FP']}FP  tau0.5 {evf['TP']}TP/{evf['FP']}FP")
        except Exception as e:
            print(f"  lib={k} {mlab:4s} ERROR {type(e).__name__}: {str(e)[:70]}")
json.dump(rows, open(os.path.join(HERE, "learning_curve.json"), "w"), indent=1)
with open(os.path.join(HERE, "learning_curve.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote learning_curve.json / .csv")
