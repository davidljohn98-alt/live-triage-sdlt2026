"""No-exploit window: paper pipeline vs one record per on-chain event; all models and rules; heuristics."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
import train_score_graph_models as TG
def npdt_alerts(s, wfrac=0.3):
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
tr=library(QC)
te,_=stream_df(full(QC))
MODELS=[("gradient_boosting","GB"),("random_forest","RF"),("extra_trees","ET"),("logistic_regression","LR"),("mlp","MLP"),("isolation_forest","IF"),("learned_graph","GNN")]
def sc(m,tr,te):
    if m=="learned_graph":
        sm=TG.learned_graph_scores(tr,te.copy())
    else:
        o=TM.fit_score(m,tr,te.copy()); sm={str(r["flow_id"]):float(r["model_score"]) for _,r in o.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])
def rules(s):
    rk=online_rank(s); return dict(max=round(float(s.max()),3),tau05=int((s>=.5).sum()),evt=len(spot(s) or []),npdt=len(npdt_alerts(s)),rank=int((rk>=.999).sum()),rank1=int((rk>=.99).sum()))
views={"paper_40732":te,
       "one_per_event_source":te[te.flow_type=="single_side_source"].reset_index(drop=True),
       "one_per_event_destination":te[te.flow_type=="single_side_destination"].reset_index(drop=True)}
res={}
for vn,v in views.items():
    res[vn]={"n":len(v),"events":int(v.flow_id.nunique()),"models":{}}
    for m,ml in MODELS:
        res[vn]["models"][ml]=rules(sc(m,tr,v))
        print(vn,ml,res[vn]["models"][ml],flush=True)
    a=amount_scores(v); rk=online_rank(a)
    res[vn]["amount_rank0.1"]=int((rk>=.999).sum()); res[vn]["amount_rank1"]=int((rk>=.99).sum()); res[vn]["route_novelty"]=len(route_novelty(v))
    print(vn,'heur',res[vn]["amount_rank0.1"],res[vn]["amount_rank1"],res[vn]["route_novelty"],flush=True)
# mean flows per day per chain for context
res["events_per_day"]=round(te.flow_id.nunique()/15,1)
json.dump(res,open(sys.argv[1]+"/step2_control.json","w"),indent=1); print("DONE")
