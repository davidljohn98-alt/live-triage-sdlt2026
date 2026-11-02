"""Saved confirmation of findings from the independent check of the camera-ready draft."""
import sys,json,math,re; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
import train_score_graph_models as TG
from collections import Counter
from datetime import datetime,timezone
ts=lambda s:int(datetime.strptime(s,"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
U=lambda x:datetime.fromtimestamp(int(x),tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
out={}
def spot_trace(s,q=1e-3,init_q=0.98):
    n=len(s); ni=max(150,int(0.15*n))
    if n<500: return None
    init=s[:ni]; t=float(np.quantile(init,init_q)); peaks=list(init[init>t]-t); p0=len(peaks); fitted=[0]
    def zq(Nt,nc):
        if len(peaks)<10: return None
        pk=np.array(peaks); m,v=pk.mean(),pk.var()
        if v<=0: fitted[0]=1; return t+m
        g=0.5*(1-m*m/v); sg=0.5*m*(m*m/v+1); r=q*nc/max(Nt,1)
        try:
            fitted[0]=1
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
    return dict(alerts=len(al),calibration_excesses=p0,tail_ever_fitted=bool(fitted[0]),idx=al)
def keyed(m,tr,v):
    if m=="learned_graph": sm=TG.learned_graph_scores(tr,v.copy())
    else:
        o=TM.fit_score(m,tr,v.copy()); sm={str(r["flow_id"]):float(r["model_score"]) for _,r in o.iterrows()}
    return np.array([sm.get(str(f),0.0) for f in v["flow_id"].astype(str)])
q,_=stream_df(full(QC)); src=q[q.flow_type=="single_side_source"].reset_index(drop=True)
tr7=library(QC)
# A two-record stream, scored record by record against keyed by flow_id
out["two_record"]={}
for m,ml in (("extra_trees","ET"),("random_forest","RF")):
    per=Scorer(m,tr7)(q); k=keyed(m,tr7,q)
    out["two_record"][ml]=dict(per_record=dict(evt=spot_trace(per)["alerts"],rank=int((online_rank(per)>=.999).sum())),keyed=dict(evt=spot_trace(k)["alerts"],rank=int((online_rank(k)>=.999).sum())),keyed_equals_row_on_source=float(np.isclose(per,k)[(q.flow_type=="single_side_source").to_numpy()].mean()),keyed_equals_row_on_destination=float(np.isclose(per,k)[(q.flow_type=="single_side_destination").to_numpy()].mean()))
print(out["two_record"],flush=True)
# B q=1e-2 on the one-record stream, and tail fitting at both risk levels
out["control_q"]={}
for m,ml in (("gradient_boosting","GB"),("random_forest","RF"),("extra_trees","ET"),("logistic_regression","LR"),("mlp","MLP"),("isolation_forest","IF"),("learned_graph","GNN")):
    s=keyed(m,tr7,src); a=spot_trace(s,1e-3); b=spot_trace(s,1e-2)
    out["control_q"][ml]=dict(q3=a["alerts"],q2=b["alerts"],excesses=a["calibration_excesses"],fitted_q3=a["tail_ever_fitted"],fitted_q2=b["tail_ever_fitted"])
print(out["control_q"],flush=True)
# C reseeds: which fits use a tail
out["control_seeds"]={}
for m,ml in (("extra_trees","ET"),("random_forest","RF"),("gradient_boosting","GB")):
    for seed in range(42,52):
        TM.RANDOM_STATE=seed; a=spot_trace(keyed(m,tr7,src))
        out["control_seeds"].setdefault(ml,[]).append(dict(seed=seed,alerts=a["alerts"],excesses=a["calibration_excesses"],fitted=a["tail_ever_fitted"]))
TM.RANDOM_STATE=42
print({k:[(x["alerts"],x["fitted"]) for x in v] for k,v in out["control_seeds"].items()},flush=True)
# D folds: tail fitting per model
out["fold_fit"]={}
for st in (ME,NO):
    tr=library(st); te,y=stream_df(full(st))
    for m,ml in (("gradient_boosting","GB"),("random_forest","RF"),("extra_trees","ET"),("logistic_regression","LR"),("mlp","MLP"),("isolation_forest","IF")):
        s=keyed(m,tr,te); a=spot_trace(s,1e-3); ix=np.array(a["idx"],dtype=int)
        out["fold_fit"].setdefault(NAMES[st],{})[ml]=dict(tp=int(y[ix].sum()) if len(ix) else 0,fp=int(len(ix)-(y[ix].sum() if len(ix) else 0)),excesses=a["calibration_excesses"],fitted=a["tail_ever_fitted"])
print(out["fold_fit"],flush=True)
# E Nomad addresses against the published list
L=pd.read_csv(GEN+"/../benchmark_release_versionA/metadata/actor_sets/nomad_2022_authentication_bypass_attackers.csv",dtype=str,keep_default_na=False) if False else None
import glob
cand=glob.glob(GEN+"/../benchmark_release_versionA/metadata/actor_sets/nomad_2022*attackers*.csv")
out["attacker_list_file"]=[c.split("/")[-1] for c in cand]
A=pd.read_csv(cand[0],dtype=str,keep_default_na=False); acol=[c for c in A.columns if "addr" in c.lower()][0]; listed=set(A[acol].str.lower())
te,y=stream_df(full(NO)); P=te[y==1]
allf=Counter(); sr=Counter(); 
for _,r in P.iterrows():
    for a_ in {str(r[c]).strip().lower() for c in ADDR if str(r[c]).strip()}:
        if a_ in listed: allf[a_]+=1
    a1=(r["src_sender"] or r["dst_recipient"]).lower()
    sr[a1]+=1
out["nomad_addresses"]=dict(list_size=len(listed),listed_in_linked_flows=len(allf),top_any_field=[(k[:12],v) for k,v in allf.most_common(4)],one_key_distinct=len(sr),one_key_listed=len([k for k in sr if k in listed]),one_key_not_listed=[k[:12] for k in sr if k not in listed],top_one_key=sr.most_common(1)[0][1],top_listed_one_key=max(v for k,v in sr.items() if k in listed))
tr=library(NO)
inlib={k:{c:int((tr[c].str.lower()==k).sum()) for c in ADDR} for k in allf if any((tr[c].str.lower()==k).any() for c in ADDR)}
out["nomad_listed_in_library"]={k[:12]:v for k,v in inlib.items()}
print(out["nomad_addresses"],out["nomad_listed_in_library"],flush=True)
# F timing with heuristics and all Nomad reports
FACTS={QU:("2022-01-27 21:34:01","2022-01-27 22:19:28"),ME:("2022-02-05 14:22:40","2022-02-05 19:31:42"),HA:("2022-06-23 11:06:46","2022-06-23 22:43:38"),NO:("2022-08-01 21:32:31","2022-08-01 21:37:23")}
out["timing_heuristics"]={}
for st in FOLDS:
    on,pub=map(ts,FACTS[st]); te,y=stream_df(full(st)); t=te["t"].astype(int).to_numpy(); d={}
    d["amount_present"]=int(((te.src_amount.str.strip()!="")|(te.dst_amount.str.strip()!="")).sum()); d["n"]=len(te)
    d["linked_before_public"]=int(((t<pub)&(y==1)).sum()); d["last_linked_to_public_min"]=round((pub-int(t[y==1].max()))/60,1)
    def first(ix):
        ix=np.array(ix,dtype=int); la=[i for i in ix if y[i]==1 and t[i]>=on]
        if not la: return dict(first=None,benign_before=None,linked=int(y[ix].sum()) if len(ix) else 0,benign=int((y[ix]==0).sum()) if len(ix) else 0)
        return dict(first=U(t[la[0]]),offset_min=round((t[la[0]]-on)/60,2),lead_on_public_min=round((pub-t[la[0]])/60,1),benign_before=int(((t[ix]<t[la[0]])&(y[ix]==0)).sum()),linked=int(y[ix].sum()),benign=int((y[ix]==0).sum()))
    d["route_novelty"]=first(route_novelty(te))
    if d["amount_present"]:
        r=online_rank(amount_scores(te)); d["amount_0.1"]=first(np.where(r>=.999)[0]); d["amount_1"]=first(np.where(r>=.99)[0])
    out["timing_heuristics"][NAMES[st]]=d
te,y=stream_df(full(NO)); t=te["t"].astype(int).to_numpy()
out["nomad_arrived_by_report"]={k:int(((t<ts(v))&(y==1)).sum()) for k,v in {"spreekaway 21:37:23":"2022-08-01 21:37:23","foobar 22:35:00":"2022-08-01 22:35:00","The Block 23:05:20":"2022-08-01 23:05:20","Nomad team 23:25:01":"2022-08-01 23:25:01","Paladin 23:48:22":"2022-08-01 23:48:22"}.items()}
print(out["timing_heuristics"],out["nomad_arrived_by_report"],flush=True)
# G library orderings: share of ET EVT recoveries on the reused address
R="0xa8c83b1b30291a3a1a118058b5445cc83041cd9d"; act=np.array([(a or b).lower() for a,b in zip(te.src_sender,te.dst_recipient)])
ORD={"Me":[ME],"Ha":[HA],"Ha,Me":[HA,ME],"Me,CS,Po,MC":[ME,CS,PO,MC],"CS,Me,Po,MC":[CS,ME,PO,MC],"CS,Po,MC,Qu,Me":[CS,PO,MC,QU,ME],"CS,Po,MC,Qu,Ha":[CS,PO,MC,QU,HA],"CS,Po,MC":[CS,PO,MC],"CS,Po,MC,Qu":[CS,PO,MC,QU],"chrono6":[CS,PO,MC,QU,ME,HA]}
out["library_reused_address"]={}
for k,o in ORD.items():
    trk=pd.concat([full(s_) for s_ in o],ignore_index=True); s=keyed("extra_trees",trk,te); al=np.array(spot_trace(s)["idx"],dtype=int); la=al[y[al]==1] if len(al) else al
    out["library_reused_address"][k]=dict(linked=len(la),benign=int(len(al)-len(la)),on_reused=int((act[la]==R).sum()),distinct=int(len(set(act[la]))))
print(out["library_reused_address"],flush=True)
# H labels on relay rows
te,y=stream_df(full(QU)); h=set(x for x in te[y==1].dst_tx_hash if x)|set(x for x in te[y==1].src_tx_hash if x); B=te[y==0]
out["qubit_benign_sharing_tx"]=[int(sum(1 for a,b in zip(B.src_tx_hash,B.dst_tx_hash) if (a in h) or (b in h))),len(B)]
te,y=stream_df(full(ME))
out["meter_payouts"]={k:[dict(label=int(y[i]),t=U(te.iloc[i]["t"]),sender=te.iloc[i]["dst_sender"][:12]) for i in range(len(te)) if te.iloc[i]["dst_tx_hash"].lower().startswith(k) or te.iloc[i]["src_tx_hash"].lower().startswith(k)] for k in ("0xd619ace8","0x519a6630")}
print(out["qubit_benign_sharing_tx"],out["meter_payouts"],flush=True)
# I amount fields and calendar span per panel
out["panels"]={}
for st in ORDER:
    te,y=stream_df(full(st)); t=te["t"].astype(int).to_numpy(); days=pd.Series(t).map(lambda x:U(x)[:10])
    out["panels"][NAMES[st]]=dict(n=len(te),amount_rows=int(((te.src_amount.str.strip()!="")|(te.dst_amount.str.strip()!="")).sum()),dates_with_flows=int(days.nunique()),calendar_days=int((datetime.strptime(days.max(),"%Y-%m-%d")-datetime.strptime(days.min(),"%Y-%m-%d")).days+1),span_days=round((t.max()-t.min())/86400,2))
print(out["panels"],flush=True)
# J Nomad: self-transfer indicator
te,y=stream_df(full(NO)); S=Scorer("extra_trees",library(NO)); F=TM.featurise(te,S.sf,S.rf)
out["self_transfer"]=dict(linked_with_indicator=int((F.sender_recipient_equal.to_numpy()[y==1]>0).sum()),benign_share=round(float(F.sender_recipient_equal.to_numpy()[y==0].mean()),3))
for nm,cols in (("zero_linked_keep_indicator",("src_sender_freq","dst_recipient_freq")),("zero_linked_reset_indicator",("src_sender_freq","dst_recipient_freq","sender_recipient_equal"))):
    G=F.copy()
    for c in cols: G.loc[y==1,c]=0.0
    s0=S.pipe.predict_proba(G)[:,1]; a0=np.where(online_rank(s0)>=.999)[0]; e0=np.array(spot(s0) or [],dtype=int)
    out["self_transfer"][nm]=dict(rank=[int(y[a0].sum()),int((y[a0]==0).sum())],evt=[int(y[e0].sum()),int((y[e0]==0).sum())])
print(out["self_transfer"],flush=True)
# K attribution shares
J=json.load(open(GEN+"/live_detection/alert_attribution.json"))
print('attr keys',list(J[0].keys())[:14],flush=True)
out["attr_sample"]={k:J[0][k] for k in list(J[0].keys())[:14]}
json.dump(out,open(sys.argv[1]+"/step14.json","w"),indent=1,default=str); print("DONE")
