import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime,timezone
ts=lambda s:int(datetime.strptime(s,"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
U=lambda x:datetime.fromtimestamp(int(x),tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
FACTS={QU:("2022-01-27 21:34:01","2022-01-27 22:19:28"),ME:("2022-02-05 14:22:40","2022-02-05 19:31:42"),HA:("2022-06-23 11:06:46","2022-06-23 22:43:38"),NO:("2022-08-01 21:32:31","2022-08-01 21:37:23")}
out={}
for st in FOLDS:
    on,pub=map(ts,FACTS[st]); tr=library(st); te,y=stream_df(full(st)); t=te["t"].astype(int).to_numpy(); s=Scorer("extra_trees",tr)(te); a=np.where(online_rank(s)>=.999)[0]
    la=[int(i) for i in a if y[i]==1]; lon=[i for i in la if t[i]>=on]; ben=[int(i) for i in a if y[i]==0]
    d=dict(onset=U(on),public=U(pub),n_linked=int(y.sum()),linked_before_public=int(((t<pub)&(y==1)).sum()),linked_before_onset=int(((t<on)&(y==1)).sum()),first_linked=U(t[y==1].min()),last_linked=U(t[y==1].max()),last_linked_to_public_min=round((pub-int(t[y==1].max()))/60,1),
      alerts_linked_pre_onset=[U(t[i]) for i in la if t[i]<on],first_linked_alert_from_onset=U(t[lon[0]]) if lon else None,offset_from_onset_min=round((t[lon[0]]-on)/60,1) if lon else None,lead_on_public_min=round((pub-t[lon[0]])/60,1) if lon else None,
      benign_before_first_from_onset=sum(1 for i in ben if lon and t[i]<t[lon[0]]),benign_total=len(ben),benign_times=[U(t[i]) for i in ben])
    if lon: d["first_from_onset_tx"]=(te.iloc[lon[0]]["src_tx_hash"][:12],te.iloc[lon[0]]["dst_tx_hash"][:12])
    out[NAMES[st]]=d; print(NAMES[st],json.dumps(d),flush=True)
json.dump(out,open(sys.argv[1]+"/step13.json","w"),indent=1)
