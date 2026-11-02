"""Figures for the camera-ready version: annotated Nomad timeline and precision-recall curves."""
import sys,json; sys.path.insert(0,sys.argv[1])
from common import *
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import datetime,timezone,timedelta
import matplotlib.dates as md
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False})
OUT=sys.argv[2]
tr=library(NO); te,y=stream_df(full(NO)); t=te["t"].astype(int).to_numpy()
s=Scorer("extra_trees",tr)(te); ev=np.array(spot(s)); pol=np.where(online_rank(s)>=.999)[0]
D=lambda x:datetime.fromtimestamp(int(x),tz=timezone.utc).replace(tzinfo=None)
X0,X1=datetime(2022,8,1),datetime(2022,8,2,6)
ON=1659389551
pos=t[y==1]; first=int(pos.min()); la=[i for i in ev if y[i]==1]; fa=int(t[la[0]])
assert round((fa-first)/3600,1)==15.1 and len(la)==13 and len(ev)==15 and len(pol)==16
fig,(ax,rug)=plt.subplots(2,1,figsize=(4.75,3.05),sharex=True,gridspec_kw=dict(height_ratios=[5,1],hspace=.05))
xs=[X0]+[D(v) for v in pos]+[X1]; ys=[0]+list(range(1,len(pos)+1))+[len(pos)]
ax.fill_between(xs,ys,step="post",color="#c9ced6",alpha=.55,lw=0); ax.step(xs,ys,where="post",color="#7f8792",lw=1.6,label="incident-linked arrivals (cumulative)")
lt=sorted(t[la]); xr=[D(lt[0])]+[D(v) for v in lt]+[X1]; yr=[0]+list(range(1,len(lt)+1))+[len(lt)]
ax.step(xr,yr,where="post",color="#c62828",lw=2,label="linked flows flagged by the extreme-value rule")
ax.plot([X1],[13],"o",color="#c62828",ms=4.5,clip_on=False); ax.annotate("13",(X1,13),xytext=(6,-3),textcoords="offset points",color="#c62828",fontweight="bold",annotation_clip=False)
ax.plot([X1],[111],"o",color="#1565c0",ms=4.5,clip_on=False); ax.annotate("111",(X1,111),xytext=(6,-3),textcoords="offset points",color="#1565c0",fontweight="bold",annotation_clip=False)
ax.plot([X1,X1],[13,111],color="#1565c0",lw=1,ls=":",label="after seeded address propagation")
ax.annotate("615",(X1,615),xytext=(6,-3),textcoords="offset points",color="#6b7280",annotation_clip=False)
# lag bracket
yb=150
ax.annotate("",xy=(D(fa),yb),xytext=(D(first),yb),arrowprops=dict(arrowstyle="<->",color="#7a5c00",lw=1.2))
ax.text(D((first+fa)/2),yb+12,"15.1 h lag, from the first linked arrival\nto the first linked alert",ha="center",va="bottom",color="#7a5c00",fontsize=7)
for v,lab,ha,dx in ((first,"first linked arrival\n04:44:51","center",0),(fa,"first linked alert\n19:50:37","right",-3)):
    ax.plot([D(v),D(v)],[0,yb],color="#7a5c00",lw=.9,ls="--")
    ax.annotate(lab,(D(v),yb),xytext=(dx,-26 if ha=="center" else -26),textcoords="offset points",ha=ha,va="center",fontsize=7.5,color="#7a5c00") if False else None
ax.text(D(first),yb-14,"first linked arrival\n04:44:51",ha="center",va="top",fontsize=6.5,color="#7a5c00",bbox=dict(fc="white",ec="none",pad=1,alpha=.85))
ax.text(D(fa-600),yb-14,"first linked alert\n19:50:37",ha="right",va="top",fontsize=6.5,color="#7a5c00",bbox=dict(fc="white",ec="none",pad=1,alpha=.85))
# onset
ax.axvline(D(ON),color="#111",lw=1.3,ls="-"); ax.text(D(ON-500),640,"exploit onset, first exploit\ntransaction 21:32:31",ha="right",va="top",fontsize=7,fontweight="bold",color="#111",bbox=dict(fc="white",ec="none",pad=1,alpha=.85))
ax.set_ylim(0,650); ax.set_ylabel("incident-linked flows"); ax.set_xlim(X0,X1); ax.grid(axis="y",color="#ededed",lw=.8); ax.set_axisbelow(True)
ax.legend(loc="center left",bbox_to_anchor=(0.0,.60),frameon=True,framealpha=.92,edgecolor="#ddd",fontsize=6.5)
for row,(idx,nm) in enumerate(((ev,"extreme-value (15)"),(pol,"policy (16)"))):
    yy=1-row
    for i in idx:
        rug.plot([D(t[i])]*2,[yy-.35,yy+.35],color="#c62828" if y[i]==1 else "#8d939b",lw=1.4)
rug.set_ylim(-.6,1.6); rug.set_yticks([1,0]); rug.set_yticklabels(["extreme-value","policy"],fontsize=6.5); rug.axvline(D(ON),color="#111",lw=1.3); rug.tick_params(axis="y",length=0); rug.spines["left"].set_visible(False)
rug.xaxis.set_major_locator(md.HourLocator(byhour=range(0,24,6))); rug.xaxis.set_major_formatter(md.DateFormatter("%H:%M")); rug.set_xlabel("observation time, 1 to 2 August 2022 (UTC); ticks: red linked, grey benign",fontsize=7)
fig.savefig(OUT+"/nomad_timeline_v2.pdf",bbox_inches="tight"); fig.savefig(OUT+"/nomad_timeline_v2.png",dpi=170,bbox_inches="tight"); plt.close(fig)
# PR curves
C=json.load(open(sys.argv[1]+"/pr_curves.json")); R=json.load(open(sys.argv[1]+"/step3.json"))
fig,axs=plt.subplots(2,2,figsize=(4.75,3.3),sharey=True,sharex=True)
fig.subplots_adjust(hspace=.3,wspace=.08)
for ax,f in zip(axs.ravel(),("Qubit","Meter","Harmony","Nomad")):
    r=R[f]
    ax.step(C[f]["ET"]["r"],C[f]["ET"]["p"],where="post",color="#c62828",lw=1.4,label="ET score")
    if f in ("Harmony","Nomad"): ax.step(C[f]["Amount"]["r"],C[f]["Amount"]["p"],where="post",color="#1565c0",lw=1.4,label="amount heuristic")
    else: ax.plot([],[],color="#1565c0",lw=1.4,label="amount heuristic")
    ax.axhline(r["base"],color="#888",lw=.9,ls="--",label="flow base rate")
    ax.set_title(f"{f} ({r['npos']})",fontsize=8.5); ax.set_xlim(0,1); ax.set_ylim(0,1.03); ax.tick_params(labelsize=7.5); ax.grid(color="#eee",lw=.6)
    ax.text(.97,.95,(f"AP {r['ET']['ap']:.2f} / {r['Amount']['ap']:.2f}" if f in ("Harmony","Nomad") else f"AP {r['ET']['ap']:.2f} / no amounts"),ha="right",va="top",fontsize=7,transform=ax.transAxes,bbox=dict(fc="white",ec="none",pad=1,alpha=.85))
axs[0,0].set_ylabel("precision",fontsize=8); axs[1,0].set_ylabel("precision",fontsize=8); axs[1,0].set_xlabel("recall",fontsize=8); axs[1,1].set_xlabel("recall",fontsize=8)
h,l=axs[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="upper center",ncol=3,fontsize=7.5,frameon=False,bbox_to_anchor=(.5,1.02))
fig.savefig(OUT+"/pr_curves.pdf",bbox_inches="tight"); fig.savefig(OUT+"/pr_curves.png",dpi=170,bbox_inches="tight"); plt.close(fig)
print("figs ok")
