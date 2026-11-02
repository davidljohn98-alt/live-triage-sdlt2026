"""Recall at fixed budgets, PR curves, exact intervals, actor-level view for the pre-specified policy."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
from scipy.stats import beta
from sklearn.metrics import average_precision_score, precision_recall_curve
def cp(k,n,a=.05):
    if n==0: return [None,None]
    lo=0.0 if k==0 else beta.ppf(a/2,k,n-k+1); hi=1.0 if k==n else beta.ppf(1-a/2,k+1,n-k)
    return [round(float(lo),3),round(float(hi),3)]
res={}; curves={}
for st in FOLDS:
    tr=library(st); te,y=stream_df(full(st)); t=te["t"].astype(int).to_numpy(); n=len(te); npos=int(y.sum())
    sE=Scorer("extra_trees",tr)(te); sA=amount_scores(te)
    d=dict(n=n,npos=npos,base=round(npos/n,4))
    rk=online_rank(sE); a=np.where(rk>=.999)[0]; tp=int(y[a].sum()); fp=len(a)-tp
    d["policy"]=dict(tp=tp,fp=fp,recall=round(tp/npos,3),recall_ci=cp(tp,npos),precision=round(tp/(tp+fp),3) if tp+fp else None,precision_ci=cp(tp,tp+fp))
    for nm,s in (("ET",sE),("Amount",sA)):
        order=np.argsort(-s,kind="stable")
        d[nm]=dict(ap=round(float(average_precision_score(y,s)),3),recall_at={str(B):round(float(y[order[:B]].sum())/npos,3) for B in (10,20,50,100)},hits_at={str(B):int(y[order[:B]].sum()) for B in (10,20,50,100)})
        pr,rc,_=precision_recall_curve(y,s); curves.setdefault(NAMES[st],{})[nm]=dict(p=[round(float(x),4) for x in pr],r=[round(float(x),4) for x in rc])
    # amount heuristic intervals
    ra=online_rank(sA); aa=np.where(ra>=.999)[0]; tpa=int(y[aa].sum()); d["amount0.1"]=dict(tp=tpa,fp=len(aa)-tpa,recall_ci=cp(tpa,npos),precision_ci=cp(tpa,len(aa)))
    a1=np.where(ra>=.99)[0]; tp1=int(y[a1].sum()); d["amount1"]=dict(tp=tp1,fp=len(a1)-tp1,recall_ci=cp(tp1,npos),precision_ci=cp(tp1,len(a1)))
    rn=route_novelty(te); tpn=int(y[rn].sum()) if rn else 0; d["route"]=dict(tp=tpn,fp=len(rn)-tpn,recall_ci=cp(tpn,npos),precision_ci=cp(tpn,len(rn)))
    # actors
    act=[(a_ or b_).lower() for a_,b_ in zip(te["src_sender"],te["dst_recipient"])]
    A=[act[i] for i in range(n) if y[i]==1]; from collections import Counter; C=Counter(A)
    d["actors"]=dict(distinct=len(C),top=C.most_common(1)[0][1],top10=sum(v for _,v in C.most_common(10)))
    La=[act[i] for i in a if y[i]==1]; d["policy_actor_hits"]=dict(distinct=len(set(La)),counts=Counter(La).most_common(5))
    if st==NO:
        al=spot(sE); Le=[act[i] for i in al if y[i]==1]; d["evt_actor_hits"]=dict(distinct=len(set(Le)),counts=Counter(Le).most_common(5))
        on=1659389551  # first exploit transaction, 21:32:31 UTC
        d["onset_ts"]=on; d["linked_before_onset"]=int(((t<on)&(y==1)).sum()); d["linked_from_onset"]=int(((t>=on)&(y==1)).sum())
        d["policy_linked_before_onset"]=int(((t[a]<on)&(y[a]==1)).sum()); d["policy_linked_from_onset"]=int(((t[a]>=on)&(y[a]==1)).sum())
        d["policy_benign_before_onset"]=int(((t[a]<on)&(y[a]==0)).sum()); d["policy_benign_from_onset"]=int(((t[a]>=on)&(y[a]==0)).sum())
        ae=np.array(al); d["evt_linked_before_onset"]=int(((t[ae]<on)&(y[ae]==1)).sum()); d["evt_linked_from_onset"]=int(((t[ae]>=on)&(y[ae]==1)).sum())
        d["evt_benign_before_onset"]=int(((t[ae]<on)&(y[ae]==0)).sum())
        d["surge_q"]=int(((t<on+3600)&(y==1)).sum())
    res[NAMES[st]]=d; print(NAMES[st],json.dumps({k:d[k] for k in ("n","npos","policy","ET","Amount","actors","policy_actor_hits")}),flush=True)
    if st==NO: print({k:v for k,v in d.items() if 'onset' in k or 'evt' in k},flush=True)
json.dump(res,open(sys.argv[1]+"/step3.json","w"),indent=1); json.dump(curves,open(sys.argv[1]+"/pr_curves.json","w")); print("DONE")
