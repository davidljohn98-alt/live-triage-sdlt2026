"""False seeds: a causal fan-out guard. An address may seed or extend propagation only while it has appeared in at most K earlier flows of the stream (K fixed at 10)."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
K=10
res={}
for st in FOLDS+(QC,):
    tr=library(st); te,y=stream_df(full(st)) if st!=QC else stream_df(full(QC).query("flow_type=='single_side_source'"))
    o=TM.fit_score("extra_trees",tr,te.copy()); sm_={str(r["flow_id"]):float(r["model_score"]) for _,r in o.iterrows()}
    s=np.array([sm_.get(str(f),0.0) for f in te["flow_id"].astype(str)]); n=len(te)
    a=np.where(online_rank(s)>=.999)[0]; AS=[addrs(te.iloc[i]) for i in range(n)]
    # seen counts up to each index
    from collections import Counter
    def run(i0):
        seen=Counter()
        for i in range(i0+1):
            seen.update(AS[i])
        S={x for x in AS[i0] if seen[x]<=K}; tp=fp=0
        for i in range(i0+1,n):
            hit=AS[i]&S
            seen.update(AS[i])
            if hit:
                tp+=int(y[i]==1); fp+=int(y[i]==0)
                S|={x for x in AS[i] if seen[x]<=K}
            S={x for x in S if seen[x]<=K}
        return tp,fp
    F=[run(int(i)) for i in a if y[i]==0]; T=[run(int(i)) for i in a if y[i]==1]
    summ=lambda L:None if not L else dict(n=len(L),flagged_mean=round(float(np.mean([x[0]+x[1] for x in L])),1),flagged_min=min(x[0]+x[1] for x in L),flagged_max=max(x[0]+x[1] for x in L),linked_mean=round(float(np.mean([x[0] for x in L])),1),linked_max=max(x[0] for x in L),benign_mean=round(float(np.mean([x[1] for x in L])),1),benign_max=max(x[1] for x in L))
    res[NAMES[st]]=dict(false=summ(F),true=summ(T)); print(NAMES[st],json.dumps(res[NAMES[st]]),flush=True)
    if st==NO:
        al=spot(s); la=[i for i in al if y[i]==1]; first=la[0]; print('Nomad first EVT seed guarded',run(first),flush=True); res[NAMES[st]]["first_evt_seed"]=run(first)
json.dump(res,open(sys.argv[1]+"/step8.json","w"),indent=1); print("DONE")
