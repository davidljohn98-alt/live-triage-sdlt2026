import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
from datetime import datetime,timezone
U=lambda x:datetime.fromtimestamp(int(x),tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
out={}; scores={}
for st in FOLDS+(QC,):
    tr=library(st); te,y=stream_df(full(st)); t=te["t"].astype(int).to_numpy(); n=len(te); npos=int(y.sum())
    sc=Scorer("extra_trees",tr)(te)
    # identity with the paper pipeline
    o=TM.fit_score("extra_trees",tr,te.copy()); m={str(r["flow_id"]):float(r["model_score"]) for _,r in o.iterrows()}
    s2=np.array([m.get(str(f),0.0) for f in te["flow_id"].astype(str)])
    same=bool(np.allclose(sc,s2))
    rL=online_rank(sc,"left"); rR=online_rank(sc,"right")
    aL=np.where(rL>=0.999)[0]; aR=np.where(rR>=0.999)[0]
    d=dict(n=n,npos=npos,identical_scores=same,rank_lt=[int(y[aL].sum()),int(len(aL)-y[aL].sum())],rank_le=[int(y[aR].sum()),int(len(aR)-y[aR].sum())])
    d["stream_start"]=U(t.min()); d["stream_end"]=U(t.max()); d["days"]=round((t.max()-t.min())/86400,2)
    if len(aL): d["first_alert_any"]=U(t[aL[0]]); d["first_alert_idx"]=int(aL[0])
    if npos:
        pos=np.where(y==1)[0]; d["first_linked"]=U(t[pos[0]]); d["last_linked"]=U(t[pos[-1]])
        d["linked_q10_q50_q90"]=[U(np.quantile(t[pos],q)) for q in (.1,.5,.9)]
        la=[i for i in aL if y[i]==1]
        if la:
            d["first_linked_alert"]=U(t[la[0]]); d["first_linked_alert_idx"]=int(la[0]); d["lag_h_from_first_linked"]=round((t[la[0]]-t[pos[0]])/3600,2)
            d["linked_arrived_by_first_linked_alert"]=int(((t<=t[la[0]])&(y==1)).sum())
        d["alerts"]=[dict(idx=int(i),t=U(t[i]),linked=int(y[i]),src=te.iloc[i]["src_chain"],dst=te.iloc[i]["dst_chain"],amt=te.iloc[i]["src_amount"] or te.iloc[i]["dst_amount"]) for i in aL]
        d["first5_linked"]=[dict(idx=int(i),t=U(t[i]),ts=te.iloc[i]["src_block_datetime_utc"],td=te.iloc[i]["dst_block_datetime_utc"],src=te.iloc[i]["src_chain"],dst=te.iloc[i]["dst_chain"],sender=te.iloc[i]["src_sender"][:10],rec=te.iloc[i]["dst_recipient"][:10],amt=te.iloc[i]["src_amount"] or te.iloc[i]["dst_amount"],rule=te.iloc[i]["linkage_rule"],lag=te.iloc[i]["lag_seconds"]) for i in pos[:5]]
    if st==NO:
        al=spot(sc); la=[i for i in al if y[i]==1]
        d["evt"]=dict(tp=len(la),fp=len(al)-len(la),first_any=U(t[al[0]]),first_linked=U(t[la[0]]),lag_h=round((t[la[0]]-t[np.where(y==1)[0][0]])/3600,2),all=[(U(t[i]),int(y[i])) for i in al])
        # hourly arrivals
        hrs=pd.Series(t[y==1]).map(lambda x:U(x)[:13]).value_counts().sort_index(); d["hourly_linked"]=hrs.to_dict()
    out[NAMES[st]]=d; scores[NAMES[st]]=sc.tolist()
    print(NAMES[st],json.dumps({k:v for k,v in d.items() if k not in("alerts","first5_linked","hourly_linked","evt")}),flush=True)
json.dump(out,open(sys.argv[1]+"/step1.json","w"),indent=1); json.dump(scores,open(sys.argv[1]+"/et_scores.json","w"))
print("DONE")
