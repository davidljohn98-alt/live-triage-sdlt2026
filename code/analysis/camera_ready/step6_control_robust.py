"""No-exploit window, one record per on-chain event (source view) against the paper's two-record stream:
EVT and rank under reseeding and training-row orderings, tree ensembles and GB."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
LIB7={"chrono":[CS,PO,MC,QU,ME,HA,NO],"meter-first":[ME,CS,PO,MC,QU,HA,NO],"meter-last":[CS,PO,MC,QU,HA,NO,ME],"reversed":[NO,HA,ME,QU,MC,PO,CS]}
te,_=stream_df(full(QC)); src=te[te.flow_type=="single_side_source"].reset_index(drop=True)
print('adjacent twins?',float((te.flow_id.values[:-1]==te.flow_id.values[1:]).mean()),flush=True)
out={"seed":{},"order":{}}
def sc(m,tr,v):
    o=TM.fit_score(m,tr,v.copy()); sm={str(r["flow_id"]):float(r["model_score"]) for _,r in o.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in v["flow_id"].astype(str)])
def rules(s):
    rk=online_rank(s); return [len(spot(s) or []),int((rk>=.999).sum())]
tr0=pd.concat([full(s) for s in LIB7["chrono"]],ignore_index=True)
for m,ml in (("extra_trees","ET"),("random_forest","RF"),("gradient_boosting","GB")):
    for seed in range(42,52):
        TM.RANDOM_STATE=seed
        a=rules(sc(m,tr0,src)); b=rules(sc(m,tr0,te))
        out["seed"].setdefault(ml,[]).append(dict(seed=seed,source=a,two_record=b))
        print(ml,seed,'source EVT,rank',a,'two-record EVT,rank',b,flush=True)
TM.RANDOM_STATE=42
for on,order in LIB7.items():
    tr=pd.concat([full(s) for s in order],ignore_index=True)
    for m,ml in (("extra_trees","ET"),("random_forest","RF"),("gradient_boosting","GB")):
        a=rules(sc(m,tr,src)); b=rules(sc(m,tr,te))
        out["order"].setdefault(ml,{})[on]=dict(source=a,two_record=b)
        print(on,ml,'source',a,'two-record',b,flush=True)
json.dump(out,open(sys.argv[1]+"/step6.json","w"),indent=1); print("DONE")
