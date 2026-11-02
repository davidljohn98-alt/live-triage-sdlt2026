#!/usr/bin/env python3
"""
Chronological rolling-origin streaming replay (headline protocol per TESSERACT
C1: every training incident strictly precedes the test incident) with a
threshold bake-off and multi-model scoring.

Folds (incident window starts verified from panel timestamps):
  Poly Network | {ChainSwap}
  Multichain   | {ChainSwap, Poly}
  Qubit        | {ChainSwap, Poly, Multichain}
  Meter        | {ChainSwap, Poly, Multichain, Qubit}
  Nomad        | {ChainSwap, Poly, Multichain, Qubit, Meter}

Models: GB / RF / LR (train_models_per_case.fit_score, canonical config,
seed 42) + GNN (train_score_graph_models.learned_graph_scores).

Threshold methods on each score stream (arrival order):
  fixed tau in {0.5, 0.9}                       (the named-pitfall baseline)
  SPOT  (EVT peaks-over-threshold, KDD 2017)    q in {1e-2, 1e-3}; skipped on
                                                 folds with < 500 flows
  NPDT  (Hundman-style nonparametric dynamic threshold, KDD 2018)
Metrics per (fold, model, method): TP, FP, precision, FPs/day, detection lag
(days, first linked ALERT); plus static AP and final-queue P@K, and budget
replay B in {10,50} per (fold, model).

Run: venv_models/bin/python live_detection/stream_chrono.py  (from generated_output)
Writes: chrono_results.json + printed summary.
"""
import sys, os, json, math
from datetime import datetime, timezone
HERE = os.path.dirname(os.path.abspath(__file__)); GEN = os.path.dirname(HERE)
sys.path.insert(0, GEN)
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
import train_models_per_case as TM
import train_score_graph_models as TG

P = os.path.join(GEN, "data/observed_panels_v2"); LBL = TM.LABEL_COL
ORDER = [  # chronological by incident window start
    ("chainswap_2021_authentication_bypass", "ChainSwap"),
    ("poly_network_2021_authentication_bypass", "Poly Network"),
    ("multichain_2022_authentication_bypass", "Multichain"),
    ("qubit_2022_bad_business_logic", "Qubit"),
    ("meter_io_2022_bad_business_logic", "Meter Passport"),
    ("harmony_horizon_2022_private_key_leak", "Harmony Horizon"),
    ("nomad_2022_authentication_bypass", "Nomad"),
]
MODELS = [("gradient_boosting","GB"), ("random_forest","RF"),
          ("logistic_regression","LR"), ("extra_trees","ET"),
          ("mlp","MLP"), ("isolation_forest","IF"), ("learned_graph","GNN")]

def load(n): return pd.read_csv(f"{P}/{n}.csv", dtype=str, keep_default_na=False)
def full(s): return pd.concat([load(f"{s}_train"), load(f"{s}_test")], ignore_index=True)
def t_obs(row):
    ts=[]
    for c in ("src_block_timestamp","dst_block_timestamp"):
        v=str(row.get(c,"")).strip()
        if v:
            try: ts.append(int(float(v)))
            except ValueError: pass
    return max(ts) if ts else None

panels = {s: full(s) for s,_ in ORDER}
starts = {}
for s,_ in ORDER:
    tt=[t_obs(r) for _,r in panels[s].iterrows()]
    starts[s]=min(t for t in tt if t is not None)
chron = sorted(ORDER, key=lambda x: starts[x[0]])
assert [c[0] for c in chron]==[c[0] for c in ORDER], f"order mismatch: {[c[1] for c in chron]}"

def scores_for(mname, tr, te):
    if mname=="learned_graph":
        sm = TG.learned_graph_scores(tr, te)
    else:
        out = TM.fit_score(mname, tr, te)
        sm = {str(r["flow_id"]): float(r["model_score"]) for _,r in out.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])

# ---------------- threshold methods ----------------
def eval_alerts(alert_idx, y, day):
    a=np.zeros(len(y),bool); a[list(alert_idx)]=True
    tp=int((a&(y==1)).sum()); fp=int((a&(y==0)).sum())
    ndays=len(set(day))
    lag=None
    if (y==1).any() and (a&(y==1)).any():
        d0=day[np.argmax(y==1)]; d1=day[np.argmax(a&(y==1))]
        lag=(datetime.fromisoformat(d1)-datetime.fromisoformat(d0)).days
    return dict(TP=tp,FP=fp,precision=tp/(tp+fp) if tp+fp else 0.0,
                fp_per_day=round(fp/ndays,2), detection_lag_days=lag, alerts=tp+fp)

def fixed_tau(s, tau): return np.where(s>=tau)[0]

def spot_alerts(s, q, n_init=None, init_q=0.98):
    """SPOT (Siffer et al. 2017), method-of-moments GPD fit on peaks."""
    n=len(s); n_init=n_init or max(150,int(0.15*n))
    if n<500: return None
    init=s[:n_init]; t=float(np.quantile(init,init_q))
    peaks=list(init[init>t]-t)
    def zq(Nt,ncur):
        if len(peaks)<10: return None
        pk=np.array(peaks); m,v=pk.mean(),pk.var()
        if v<=0: return t+m
        gamma=0.5*(1-m*m/v); sigma=0.5*m*(m*m/v+1)   # MoM GPD
        r=q*ncur/max(Nt,1)
        try:
            if abs(gamma)<1e-6: return t - sigma*math.log(r)
            return t + (sigma/gamma)*((r**(-gamma))-1)
        except (OverflowError, ValueError): return None
    z=zq(len(peaks),n_init) or float(np.quantile(init,0.999))
    alerts=[]; Nt=len(peaks); ncur=n_init
    for i in range(n_init,n):
        x=s[i]; ncur+=1
        if x>z: alerts.append(i)                 # anomaly: do not learn
        elif x>t:
            peaks.append(x-t); Nt+=1
            z2=zq(Nt,ncur)
            if z2 is not None: z=z2
    return alerts

def npdt_alerts(s, wfrac=0.3):
    """Hundman-style nonparametric dynamic threshold, sliding windows."""
    n=len(s); w=max(200,int(wfrac*n)); step=max(50,w//2)
    alerts=set()
    for lo in range(0,n,step):
        win=s[lo:lo+w]
        if len(win)<50: break
        mu,sd=float(win.mean()),float(win.std())
        if sd==0: continue
        best=None;best_eps=None
        for z in (2,2.5,3,4,5,7,10):
            eps=mu+z*sd; above=win>eps
            na=int(above.sum())
            if na==0: continue
            dmu=mu-float(win[~above].mean()); dsd=sd-float(win[~above].std())
            segs=int(np.diff(np.where(above)[0]).clip(min=0).astype(bool).size) or 1
            score=(dmu/mu if mu else 0)+(dsd/sd if sd else 0); score/= (na+segs**2)
            if best is None or score>best: best,best_eps=score,eps
        if best_eps is not None:
            for j in np.where(win>best_eps)[0]: alerts.add(lo+j)
    return sorted(alerts)

# ---------------- run ----------------
results={}
for k in range(1,len(chron)):
    stem,label=chron[k]
    tr=pd.concat([panels[s] for s,_ in chron[:k]],ignore_index=True)
    te=panels[stem].copy()
    te["t"]=[t_obs(r) for _,r in te.iterrows()]
    te=te[te["t"].notna()].sort_values("t",kind="stable").reset_index(drop=True)
    y=(te[LBL].astype(str)=="1").astype(int).to_numpy()
    t=te["t"].astype(int).to_numpy()
    day=np.array([datetime.fromtimestamp(int(x),tz=timezone.utc).date().isoformat() for x in t])
    npos=int(y.sum()); base=npos/len(te)
    fold=dict(label=label,n=len(te),pos=npos,base=round(base,4),
              trained_on=[c[1] for c in chron[:k]],models={})
    print(f"\n{'='*90}\n{label}: {len(te):,} flows, {npos} linked ({base:.2%}) | trained on {fold['trained_on']}")
    for mname,mlab in MODELS:
        try: s=scores_for(mname,tr,te.copy())
        except Exception as e:
            print(f"  {mlab}: ERROR {e}"); continue
        ap=float(average_precision_score(y,s)) if npos else float("nan")
        order=np.argsort(-s,kind="stable")
        topk={K:dict(hits=int(y[order[:K]].sum()),precision=float(y[order[:K]].mean()))
              for K in (10,50,100) if K<=len(y)}
        # budget replay
        replay={}
        for B in (10,50):
            reviewed=np.zeros(len(te),bool); found=0; first=None
            for d in sorted(set(day)):
                arr=np.where((day<=d)&~reviewed)[0]
                if len(arr):
                    pick=arr[np.argsort(-s[arr],kind="stable")[:B]]
                    reviewed[pick]=True; got=int(y[pick].sum()); found+=got
                    if got and first is None: first=d
            lag=None
            if first and npos:
                d0=day[np.argmax(y==1)]
                lag=(datetime.fromisoformat(first)-datetime.fromisoformat(d0)).days
            replay[B]=dict(found=found,recall=round(found/npos,3) if npos else None,
                           reviewed=int(reviewed.sum()),lag=lag)
        # thresholds
        thr={}
        thr["tau0.5"]=eval_alerts(fixed_tau(s,0.5),y,day)
        thr["tau0.9"]=eval_alerts(fixed_tau(s,0.9),y,day)
        for q in (1e-2,1e-3):
            al=spot_alerts(s,q)
            thr[f"SPOT q={q:g}"]=eval_alerts(al,y,day) if al is not None else "skipped (n<500)"
        thr["NPDT"]=eval_alerts(npdt_alerts(s),y,day)
        fold["models"][mlab]=dict(ap=round(ap,4),topk=topk,replay=replay,thresholds=thr)
        tl=" | ".join(f"{k2}: TP{v['TP']}/FP{v['FP']}" for k2,v in thr.items() if isinstance(v,dict))
        print(f"  {mlab}: AP={ap:.3f}  P@10={topk.get(10,{}).get('precision',0):.0%}  "
              f"B50 found {replay[50]['found']}/{npos}  || {tl}")
    results[stem]=fold
json.dump(results,open(os.path.join(HERE,"chrono_results.json"),"w"),indent=2)
print("\nwrote chrono_results.json")
