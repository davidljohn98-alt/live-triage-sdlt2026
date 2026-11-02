"""False first alerts seeding address propagation: spread from every benign alert of the policy; guards."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
def run(te,y,seed_i,mode,cap=None):
    t=te["t"].astype(int).to_numpy(); S0=set(addrs(te.iloc[seed_i])); S=set(S0); tp=fp=0
    for i in ADJ[seed_i:] if False else range(seed_i+1,len(te)):
        A=AS[i]
        if A&S:
            tp+=int(y[i]==1); fp+=int(y[i]==0)
            if mode=="prop": S|=A
            if cap and tp+fp>=cap: break
    return tp,fp,len(S)
ADJ=None
res={}
for st in FOLDS+(QC,):
    tr=library(st); te,y=stream_df(full(st)) if st!=QC else stream_df(full(QC).query("flow_type=='single_side_source'"))
    o=TM.fit_score("extra_trees",tr,te.copy()); sm={str(r["flow_id"]):float(r["model_score"]) for _,r in o.iterrows()}
    s=np.array([sm.get(str(f),0.0) for f in te["flow_id"].astype(str)])
    rk=online_rank(s); a=np.where(rk>=.999)[0]; n=len(te)
    AS=[addrs(te.iloc[i]) for i in range(n)]
    rows=[]
    for i in a:
        if y[i]==0:
            P=run(te,y,int(i),"prop"); Q=run(te,y,int(i),"seed"); C=run(te,y,int(i),"prop",cap=10)
            rows.append(dict(i=int(i),prop=P,seed=Q,cap=C))
    trues=[]
    for i in a:
        if y[i]==1:
            P=run(te,y,int(i),"prop"); Q=run(te,y,int(i),"seed"); C=run(te,y,int(i),"prop",cap=10)
            trues.append(dict(i=int(i),prop=P,seed=Q,cap=C))
    f=lambda L,k:[int(x[k][0])+int(x[k][1]) for x in L]
    d=dict(n=n,npos=int(y.sum()),false_alerts=len(rows),true_alerts=len(trues))
    for nm,L in(("false",rows),("true",trues)):
        if L:
            d[nm]=dict(prop_flagged_mean=round(float(np.mean(f(L,"prop"))),1),prop_flagged_min=min(f(L,"prop")),prop_flagged_max=max(f(L,"prop")),
                       prop_linked_mean=round(float(np.mean([x["prop"][0] for x in L])),1),prop_benign_mean=round(float(np.mean([x["prop"][1] for x in L])),1),prop_benign_max=max(x["prop"][1] for x in L),prop_linked_max=max(x["prop"][0] for x in L),
                       seed_flagged_mean=round(float(np.mean(f(L,"seed"))),1),seed_flagged_max=max(f(L,"seed")),seed_benign_max=max(x["seed"][1] for x in L),
                       cap_flagged_mean=round(float(np.mean(f(L,"cap"))),1),cap_benign_mean=round(float(np.mean([x["cap"][1] for x in L])),1),share_stream_max=round(max(f(L,"prop"))/n,4),nzero=sum(1 for x in f(L,"prop") if x==0))
    d["rows"]=rows[:40]; d["trues"]=trues[:20]
    res[NAMES[st]]=d; print(NAMES[st],json.dumps({k:v for k,v in d.items() if k not in("rows","trues")}),flush=True)
json.dump(res,open(sys.argv[1]+"/step5.json","w"),indent=1); print("DONE")
