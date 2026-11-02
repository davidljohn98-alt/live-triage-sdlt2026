import sys,json,re; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
from collections import Counter
exec(re.search(r"def npdt_alerts.*?return sorted\(alerts\)\n",open(sys.argv[1]+"/step2_control.py").read(),re.S).group(0))
out={}
# 1. does the Nomad first mover have history in the library, and with which labels
A="0xa8c83b1b30291a3a1a118058b5445cc83041cd9d"
tr=library(NO); yt=(tr[TM.LABEL_COL].astype(str)=="1").astype(int)
for c in ("src_sender","dst_recipient","src_recipient","dst_sender"):
    m=tr[c].str.lower()==A; out[c]=dict(n=int(m.sum()),linked=int(yt[m].sum()),cases=tr[m].case_id.value_counts().to_dict())
sf,rf=TM.build_frequency_maps(tr); out["freq"]=dict(sender=sf.get(TM._norm_addr(A),0),recipient=rf.get(TM._norm_addr(A),0))
# overlap of all Nomad attacker addresses with the library
te,y=stream_df(full(NO)); P=te[y==1]
aa=set(a.lower() for a in P.src_sender if a)|set(a.lower() for a in P.dst_recipient if a)
libS=set(a.lower() for a in tr.src_sender if a)|set(a.lower() for a in tr.dst_recipient if a)
out["nomad_attackers_in_library"]=dict(total=len(aa),in_library=len(aa&libS))
ov=[a for a in aa&libS]; X=[]
for a in ov[:400]:
    X.append((a[:10],sf.get(TM._norm_addr(a),0),rf.get(TM._norm_addr(a),0),int((P.src_sender.str.lower()==a).sum()+(P.dst_recipient.str.lower()==a).sum())))
out["overlap_top"]=sorted(X,key=lambda x:-(x[1]+x[2]))[:8]
# feature view of the 9 policy linked alerts vs other linked
S=Scorer("extra_trees",tr); s=S(te); a=np.where(online_rank(s)>=.999)[0]
F=TM.featurise(te,S.sf,S.rf)
la=[i for i in a if y[i]==1]
out["alerted_features"]=F.iloc[la][["route","src_sender_freq","dst_recipient_freq","sender_recipient_equal","log_src_amount","log_dst_amount","log_lag_seconds"]].round(2).to_dict("records")
oth=F[(y==1)]; out["linked_means"]=oth[["src_sender_freq","dst_recipient_freq","sender_recipient_equal"]].mean().round(3).to_dict(); out["linked_freq_pos"]=int(((oth.src_sender_freq>0)|(oth.dst_recipient_freq>0)).sum())
ben=F[(y==0)]; out["benign_means"]=ben[["src_sender_freq","dst_recipient_freq","sender_recipient_equal"]].mean().round(3).to_dict(); out["benign_freq_pos"]=int(((ben.src_sender_freq>0)|(ben.dst_recipient_freq>0)).sum())
out["scores_alerted"]=[round(float(s[i]),3) for i in la]
# 2. what shared addresses drive false-seed spread
for st in (ME,HA,NO):
    t2,y2=stream_df(full(st)); c=Counter()
    for _,r in t2.iterrows(): c.update(addrs(r))
    out["top_shared_"+NAMES[st]]=[(k[:12],v) for k,v in c.most_common(4)]
# 3. NPDT across fits on the one-record stream, and the flood run
q,_=stream_df(full(QC)); src=q[q.flow_type=="single_side_source"].reset_index(drop=True)
def sc(m,tr7,v):
    o=TM.fit_score(m,tr7,v.copy()); sm={str(r["flow_id"]):float(r["model_score"]) for _,r in o.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in v["flow_id"].astype(str)])
LIB7={"chrono":[CS,PO,MC,QU,ME,HA,NO],"meter-first":[ME,CS,PO,MC,QU,HA,NO],"meter-last":[CS,PO,MC,QU,HA,NO,ME],"reversed":[NO,HA,ME,QU,MC,PO,CS],"nomad-first":[NO,CS,PO,MC,QU,ME,HA]}
tr7=pd.concat([full(s_) for s_ in LIB7["chrono"]],ignore_index=True); res={}
for m_,ml in (("extra_trees","ET"),("random_forest","RF"),("gradient_boosting","GB")):
    v=[]
    for seed in range(42,52):
        TM.RANDOM_STATE=seed; ss=sc(m_,tr7,src); v.append(len(npdt_alerts(ss)))
        if ml=="ET" and seed==46:
            al=np.array(spot(ss)); gaps=np.where(np.diff(al)>1)[0]; st_=al[gaps[-1]+1] if len(gaps) else al[0]; out["flood_run"]=dict(start=int(st_),length=int(len(al)-(gaps[-1]+1 if len(gaps) else 0)),end=int(al[-1]),n=len(ss),plateau=float(pd.Series(ss).mode()[0]),alerted_below_plateau=int((ss[al]<pd.Series(ss).mode()[0]).sum()))
    res[ml]=v
TM.RANDOM_STATE=42
for on,o in LIB7.items():
    t7=pd.concat([full(s_) for s_ in o],ignore_index=True)
    res.setdefault("order",{})[on]={ml:len(npdt_alerts(sc(m_,t7,src))) for m_,ml in (("extra_trees","ET"),("random_forest","RF"),("gradient_boosting","GB"))}
out["npdt"]=res
json.dump(out,open(sys.argv[1]+"/step11.json","w"),indent=1,default=str); print(json.dumps(out,default=str)[:5500])
