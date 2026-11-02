import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
out={}
for st in FOLDS:
    tr=library(st); te,y=stream_df(full(st)); S=Scorer("extra_trees",tr); F=TM.featurise(te,S.sf,S.rf); s=S(te); a=np.where(online_rank(s)>=.999)[0]
    fp=((F.src_sender_freq>0)|(F.dst_recipient_freq>0)).to_numpy()
    ytr=(tr[TM.LABEL_COL].astype(str)=="1").to_numpy()
    d=dict(linked=int(y.sum()),linked_freq_pos=int((fp&(y==1)).sum()),benign=int((y==0).sum()),benign_freq_pos=int((fp&(y==0)).sum()),
           alert_linked=int(y[a].sum()),alert_linked_freq_pos=int((fp[a]&(y[a]==1)).sum()),alert_benign=int((y[a]==0).sum()),alert_benign_freq_pos=int((fp[a]&(y[a]==0)).sum()))
    if st==NO:
        al=np.array(spot(s)); d["evt_linked_freq_pos"]=int((fp[al]&(y[al]==1)).sum()); d["evt_linked"]=int(y[al].sum()); d["evt_benign_freq_pos"]=int((fp[al]&(y[al]==0)).sum())
    # zeroing the frequency features only, everything else unchanged
    pipe=S.pipe; G=F.copy(); G["src_sender_freq"]=0.0; G["dst_recipient_freq"]=0.0
    s0=pipe.predict_proba(G)[:,1]; a0=np.where(online_rank(s0)>=.999)[0]; d["zero_freq_rank"]=[int(y[a0].sum()),int((y[a0]==0).sum())]
    if st==NO:
        e0=spot(s0) or []; e0=np.array(e0,dtype=int); d["zero_freq_evt"]=[int(y[e0].sum()),int((y[e0]==0).sum())]
    out[NAMES[st]]=d; print(NAMES[st],d,flush=True)
json.dump(out,open(sys.argv[1]+"/step12.json","w"),indent=1)
