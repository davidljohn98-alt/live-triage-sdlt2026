#!/usr/bin/env python3
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=json.load(open("learning_curve.json"))
OUT="figures_timeseries"; os.makedirs(OUT,exist_ok=True)
models=sorted({r["model"] for r in rows})
sizes=sorted({r["library_size"] for r in rows})
COL={"GB":"#c62828","RF":"#1565c0","LR":"#6a1b9a","ET":"#2e7d32","MLP":"#ef6c00","GNN":"#00838f","IF":"#616161"}
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
# left: recovered linked flows under the committed EVT policy
for m in models:
    ys=[next((r["evt_tp"] for r in rows if r["library_size"]==k and r["model"]==m),0) for k in sizes]
    axes[0].plot(sizes,ys,marker="o",lw=2,color=COL.get(m,"#555"),label=m)
axes[0].set_xlabel("training library size (prior exploits)")
axes[0].set_ylabel("incident-linked flows recovered")
axes[0].set_title("Pre-specified extreme-value policy (EVT, $q{=}10^{-3}$)",fontsize=11)
axes[0].axvspan(4.5,6.5,color="#2e7d32",alpha=0.07)
axes[0].annotate("supervised transition",xy=(4.6,11.2),fontsize=9,color="#2e7d32")
# right: precision at 10
for m in models:
    ys=[100*next((r["p_at_10"] for r in rows if r["library_size"]==k and r["model"]==m),0) for k in sizes]
    axes[1].plot(sizes,ys,marker="s",lw=2,color=COL.get(m,"#555"),label=m)
axes[1].set_xlabel("training library size (prior exploits)")
axes[1].set_ylabel("precision at 10 (%)"); axes[1].set_yticks(range(0,101,20)); axes[1].set_ylim(-4,104)
axes[1].set_title("Ranking quality (precision at ten)",fontsize=11)
axes[1].set_ylim(-4,100); axes[1].set_yticks(range(0,101,20))
for ax in axes:
    ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=0.25); ax.set_xticks(sizes)
axes[0].legend(fontsize=8,ncol=3)
fig.tight_layout()
fig.savefig(f"{OUT}/F_learning_curve.png",dpi=170); fig.savefig(f"{OUT}/F_learning_curve.pdf")
print("wrote F_learning_curve.png/.pdf")
