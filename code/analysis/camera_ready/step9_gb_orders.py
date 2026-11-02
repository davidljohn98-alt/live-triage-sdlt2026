"""GB on the one-record stream under the paper's four library orderings, and the mechanism of the ET flood."""
import sys,json,re; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
exec(re.search(r"def npdt_alerts.*?return sorted\(alerts\)\n",open(sys.argv[1]+"/step2_control.py").read(),re.S).group(0))
LIB7={"chrono":[CS,PO,MC,QU,ME,HA,NO],"rev":[NO,HA,ME,QU,MC,PO,CS],"meter-first":[ME,CS,PO,MC,QU,HA,NO],"nomad-first":[NO,CS,PO,MC,QU,ME,HA]}
te,_=stream_df(full(QC)); src=te[te.flow_type=="single_side_source"].reset_index(drop=True)
def sc(m,tr,v):
    o=TM.fit_score(m,tr,v.copy()); sm={str(r["flow_id"]):float(r["model_score"]) for _,r in o.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in v["flow_id"].astype(str)])
out={"gb_orders":{}}
for on,order in LIB7.items():
    tr=pd.concat([full(s) for s in order],ignore_index=True); s=sc("gradient_boosting",tr,src); rk=online_rank(s)
    out["gb_orders"][on]=dict(evt=len(spot(s) or []),npdt=len(npdt_alerts(s)),rank=int((rk>=.999).sum()),max=round(float(s.max()),3)); print(on,out["gb_orders"][on],flush=True)
tr=pd.concat([full(s) for s in LIB7["chrono"]],ignore_index=True)
for seed in (42,46):
    TM.RANDOM_STATE=seed; s=sc("extra_trees",tr,src); al=spot(s) or []; n=len(s)
    ni=max(150,int(.15*n)); init=s[:ni]; u=float(np.quantile(init,.98))
    info=dict(seed=seed,alerts=len(al),first=al[0] if al else None,n=n,ni=ni,u=u,exceed_u_in_init=int((init>u).sum()),distinct_init=int(len(np.unique(init))),distinct_all=int(len(np.unique(s))),top_values=[(round(float(v),5),int(c)) for v,c in sorted(zip(*np.unique(s,return_counts=True)),key=lambda x:-x[1])[:5]])
    if al:
        a=np.array(al); info["alert_span"]=[int(a[0]),int(a[-1])]; info["distinct_alerted"]=int(len(np.unique(s[a]))); info["min_alerted"]=float(s[a].min())
        info["share_after_first_alert"]=round(len(al)/(n-a[0]),3)
    out[f"et_seed{seed}"]=info; print(info,flush=True)
TM.RANDOM_STATE=42
json.dump(out,open(sys.argv[1]+"/step9.json","w"),indent=1); print("DONE")
