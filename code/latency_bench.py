#!/usr/bin/env python3
"""Transaction-to-score latency .

Measures per-stage wall-clock cost of the deployed path on real panels:
  featurise  - build the feature row for an arriving flow
  transform  - apply the fitted preprocessor
  score      - model inference
  threshold  - one online update of the adaptive alert level
Reported per flow, and against per-chain block time, which is the bound that
matters for a live claim. Writes latency_results.json.
"""
import sys, os, json, time, statistics
HERE=os.path.dirname(os.path.abspath(__file__)); GEN=os.path.dirname(HERE)
sys.path.insert(0,GEN); sys.path.insert(0,HERE)
import numpy as np, pandas as pd
from stream_chrono import panels, ORDER
import train_models_per_case as TM

stems=[s for s,_ in ORDER]
tr=pd.concat([panels[s] for s in stems[:-1]],ignore_index=True)
te=panels[stems[-1]].copy()
ytr=(tr[TM.LABEL_COL].astype(str)=="1").astype(int).to_numpy()
fm=TM.build_frequency_maps(tr)
Xtr=TM.featurise(tr,*fm)
pre=TM._build_preprocessor(); Atr=pre.fit_transform(Xtr)
Atr=Atr.toarray() if hasattr(Atr,"toarray") else Atr

res={}
for mname,mlab in (("extra_trees","ET"),("gradient_boosting","GB"),("random_forest","RF"),
                   ("logistic_regression","LR"),("mlp","MLP"),("isolation_forest","IF")):
    est,supervised=TM.make_model(mname)
    est.fit(Atr,ytr) if supervised else est.fit(Atr)
    # per-flow timings over N single-row invocations (the deployed path)
    N=200
    sub=te.head(N).reset_index(drop=True)
    t_feat=[];t_tr=[];t_sc=[]
    for i in range(N):
        row=sub.iloc[[i]]
        t0=time.perf_counter(); Xr=TM.featurise(row,*fm); t1=time.perf_counter()
        Ar=pre.transform(Xr); Ar=Ar.toarray() if hasattr(Ar,"toarray") else Ar; t2=time.perf_counter()
        (est.predict_proba(Ar) if supervised else est.score_samples(Ar)); t3=time.perf_counter()
        t_feat.append((t1-t0)*1000); t_tr.append((t2-t1)*1000); t_sc.append((t3-t2)*1000)
    tot=[a+b+c for a,b,c in zip(t_feat,t_tr,t_sc)]
    res[mlab]=dict(featurise_ms=round(statistics.median(t_feat),3),
                   transform_ms=round(statistics.median(t_tr),3),
                   score_ms=round(statistics.median(t_sc),3),
                   total_median_ms=round(statistics.median(tot),3),
                   total_p95_ms=round(sorted(tot)[int(0.95*len(tot))],3),
                   flows_per_second=round(1000/statistics.median(tot),1))
    print(f"  {mlab:3s} featurise {res[mlab]['featurise_ms']:6.2f}ms  transform {res[mlab]['transform_ms']:6.2f}ms  "
          f"score {res[mlab]['score_ms']:6.2f}ms  total median {res[mlab]['total_median_ms']:6.2f}ms "
          f"(p95 {res[mlab]['total_p95_ms']:.2f})  -> {res[mlab]['flows_per_second']:,} flows/s")

# adaptive threshold update cost
import bisect, math
seen=[]; t0=time.perf_counter()
for x in np.random.RandomState(0).rand(5000): bisect.insort(seen,float(x))
t1=time.perf_counter()
res["threshold_update_us"]=round((t1-t0)/5000*1e6,2)
print(f"  adaptive threshold update: {res['threshold_update_us']} microseconds per flow")
BLOCK_TIME={"ethereum":12.0,"bsc":3.0,"polygon":2.0,"arbitrum":0.25,"harmony":2.0,"moonbeam":12.0}
res["block_times_s"]=BLOCK_TIME
best=min(v["total_median_ms"] for k,v in res.items() if isinstance(v,dict) and "total_median_ms" in v)
print(f"  fastest scorer median {best} ms vs fastest chain block time {min(BLOCK_TIME.values())*1000:.0f} ms")
json.dump(res,open(os.path.join(HERE,"latency_results.json"),"w"),indent=1)
print("wrote latency_results.json")
