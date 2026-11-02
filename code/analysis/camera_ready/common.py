"""Shared helpers for the SDLT camera-ready analyses; same definitions as the artefact's analysis scripts."""
import sys, math, bisect, os
GEN="<ARTEFACT_ROOT>"
sys.path.insert(0,GEN)
import numpy as np, pandas as pd
import train_models_per_case as TM
from sklearn.pipeline import Pipeline
P=GEN+"/data/observed_panels_v2"; PN=GEN+"/data/observed_panels"
CS,PO,MC,QU,ME,HA,NO=("chainswap_2021_authentication_bypass","poly_network_2021_authentication_bypass",
 "multichain_2022_authentication_bypass","qubit_2022_bad_business_logic",
 "meter_io_2022_bad_business_logic","harmony_horizon_2022_private_key_leak",
 "nomad_2022_authentication_bypass")
QC="socket_2024_quiet_control"
ORDER=[CS,PO,MC,QU,ME,HA,NO]
NAMES={CS:"ChainSwap",PO:"Poly",MC:"Multichain",QU:"Qubit",ME:"Meter",HA:"Harmony",NO:"Nomad",QC:"NoExploit"}
LIBS={QU:(CS,PO,MC),ME:(CS,PO,MC,QU),HA:(CS,PO,MC,QU,ME),NO:(CS,PO,MC,QU,ME,HA),QC:tuple(ORDER)}
FOLDS=(QU,ME,HA,NO)
ADDR=("src_sender","src_recipient","dst_sender","dst_recipient")
def full(s,base=None):
    b=base or (PN if s==QC else P)
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
def online_rank(s,side="left"):
    out=np.zeros(len(s)); seen=[]
    f=bisect.bisect_left if side=="left" else bisect.bisect_right
    for i,x in enumerate(s):
        out[i]=f(seen,x)/len(seen) if seen else 0.0
        bisect.insort(seen,float(x))
    return out
class Scorer:
    """Fit once on the library, then score any number of perturbed streams; identical to TM.fit_score for the first stream."""
    def __init__(self,mname,tr):
        self.sf,self.rf=TM.build_frequency_maps(tr)
        X=TM.featurise(tr,self.sf,self.rf)
        est,sup=TM.make_model(mname); self.sup=sup
        self.pipe=Pipeline([("pre",TM._build_preprocessor()),("clf",est)])
        if sup:
            yy=(tr.get(TM.LABEL_COL,"0").astype(str)=="1").astype(int).to_numpy(); self.pipe.fit(X,yy)
        else: self.pipe.fit(X)
    def __call__(self,te):
        X=TM.featurise(te,self.sf,self.rf)
        if self.sup: return self.pipe.predict_proba(X)[:,1]
        return -self.pipe.named_steps["clf"].score_samples(self.pipe.named_steps["pre"].transform(X))
def library(stem): return pd.concat([full(s) for s in LIBS[stem]],ignore_index=True)
def addrs(row): return {str(row.get(c,"")).strip().lower() for c in ADDR if str(row.get(c,"")).strip()}
def amount_scores(te):
    amt=np.zeros(len(te))
    for i,(_,r) in enumerate(te.iterrows()):
        v=0.0
        for c in ("src_amount_usd","dst_amount_usd","src_amount","dst_amount"):
            x=str(r.get(c,"")).strip()
            if x:
                try: v=max(v,abs(float(x)))
                except ValueError: pass
        amt[i]=np.log1p(v)
    return amt
def route_novelty(te):
    warm=int(0.2*len(te)); seen=set(); alerts=[]
    cols=[c for c in ("bridge","src_chain","dst_chain","route") if c in te.columns]
    for i,(_,r) in enumerate(te.iterrows()):
        key=tuple(str(r.get(c,"")) for c in cols)
        if key not in seen:
            if i>=warm: alerts.append(i)
            seen.add(key)
    return alerts
