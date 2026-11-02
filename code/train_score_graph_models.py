#!/usr/bin/env python3
"""Stage 6 (graph): Graph-based baseline fitting and ranked scoring.

Implements the three graph-based baseline classes of Section 4.7:

  (ii)  pagerank             -- Eigenvector-based graph-ranking (Section 4.7.2).
                                Directed actor transfer graph induced by the
                                COMPLETE reconstructed training flows
                                (src_sender -> dst_recipient); damped PageRank
                                stationary vector; flow score = q[a_s] + q[a_d].
  (iii) learned_graph        -- Learned graph-structured baseline (Section 4.7.3).
                                Heterogeneous actor / route-context graph; K-hop
                                symmetric-normalised feature propagation
                                (SGC-style graph convolution, Wu et al. 2019);
                                MLP readout over the two endpoint embeddings,
                                trained on the training-partition labels.
  (iv)  cross_chain_specific -- Cross-chain-specific graph-mining baseline
                                (Section 4.7.4). Seeded personalised-PageRank
                                (guilt-by-association) over the cross-chain
                                actor / route-context graph, seeded on the
                                training-incident actors; flow score = r[a_s]+r[a_d].

The driver mirrors train_models_per_case.py: a GLOBAL (pooled) regime and a
PER-CASE regime, scored under the identical ranked-evaluation metric
(Average Precision, P@K, R@K).  No torch dependency -- NumPy + scipy.sparse
+ scikit-learn only.

Run (project ML venv):
    venv_models/bin/python3 train_score_graph_models.py \
        --panels-subdir observed_panels --scores-subdir scores_observed_graph \
        --regime both
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

RANDOM_STATE = 42
LABEL_COL = "incident_linked"
FLOW_ID_COL = "flow_id"
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Address / actor helpers
# ---------------------------------------------------------------------------

def _norm(x: Any) -> str:
    return "" if x is None else str(x).strip().lower()


def src_actor(r: Dict[str, str]) -> str:
    """Source-side actor: sender, else source-side recipient."""
    return _norm(r.get("src_sender")) or _norm(r.get("src_recipient"))


def dst_actor(r: Dict[str, str]) -> str:
    """Destination-side actor: recipient, else destination-side sender."""
    return _norm(r.get("dst_recipient")) or _norm(r.get("dst_sender"))


def _to_float(x: Any) -> float:
    try:
        v = float(x)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return 0.0


def _context_key(r: Dict[str, str]) -> str:
    """Route / bridge context node key for a flow."""
    sc = _norm(r.get("src_chain")); dc = _norm(r.get("dst_chain"))
    br = _norm(r.get("affected_bridge_protocol"))
    return f"ctx::{sc}>{dc}|{br}"


# ---------------------------------------------------------------------------
# Ranked metrics (identical formulae to ranked_evaluation.py / Eqs. 19-20)
# ---------------------------------------------------------------------------

def ranked_metrics(scores: Dict[str, float], labels: Dict[str, int],
                   k_cutoffs=(10, 25, 50, 100, 250, 500)) -> Dict[str, Any]:
    pairs = [(scores.get(fid, 0.0), y, fid) for fid, y in labels.items()]
    pairs.sort(key=lambda t: (-t[0], t[2]))
    n = len(pairs)
    total_pos = sum(y for _s, y, _f in pairs)
    out: Dict[str, Any] = {"n_test": n, "n_test_pos": total_pos}
    cum = 0; ap = 0.0; prev_r = 0.0
    for i, (_s, y, _f) in enumerate(pairs, start=1):
        cum += y
        if i in k_cutoffs:
            out[f"P@{i}"] = round(cum / i, 6)
            out[f"R@{i}"] = round((cum / total_pos) if total_pos else 0.0, 6)
        if total_pos:
            p_i = cum / i; r_i = cum / total_pos
            ap += (r_i - prev_r) * p_i; prev_r = r_i
    out["average_precision"] = round(ap, 6)
    return out


# ===========================================================================
# (ii) PageRank -- directed actor transfer graph from COMPLETE flows
# ===========================================================================

def pagerank_scores(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    alpha: float = 0.15, max_iter: int = 200,
                    tol: float = 1e-12) -> Dict[str, float]:
    # Edges induced by complete flows only: a_s -> a_d.
    idx: Dict[str, int] = {}
    rows: List[int] = []; cols: List[int] = []; vals: List[float] = []

    def node(a: str) -> int:
        if a not in idx:
            idx[a] = len(idx)
        return idx[a]

    is_complete = train_df.get("flow_type", pd.Series(dtype=str)).astype(str) == "complete"
    for _, r in train_df[is_complete].iterrows():
        a_s = src_actor(r); a_d = dst_actor(r)
        if not a_s or not a_d:
            continue
        i = node(a_d); j = node(a_s)        # edge j -> i (in-edge of i)
        rows.append(i); cols.append(j); vals.append(1.0)

    N = len(idx)
    if N == 0:
        return {r["flow_id"]: 0.0 for _, r in test_df.iterrows()}
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    out_deg = np.asarray(A.sum(axis=0)).ravel()          # column sums = out-strength
    dangling = (out_deg == 0.0)
    inv = np.zeros(N); nz = ~dangling; inv[nz] = 1.0 / out_deg[nz]
    M = A @ sp.diags(inv)                                 # column-stochastic on non-dangling

    q = np.full(N, 1.0 / N)
    teleport = np.full(N, 1.0 / N)
    for _ in range(max_iter):
        dmass = q[dangling].sum()
        new = (1.0 - alpha) * (M @ q + dmass / N) + alpha * teleport
        s = new.sum()
        if s > 0:
            new /= s
        if np.abs(new - q).sum() < tol:
            q = new; break
        q = new

    qd = {a: float(q[i]) for a, i in idx.items()}
    out: Dict[str, float] = {}
    for _, r in test_df.iterrows():
        out[r["flow_id"]] = qd.get(src_actor(r), 0.0) + qd.get(dst_actor(r), 0.0)
    return out


# ===========================================================================
# Heterogeneous actor / route-context graph (shared by iii and iv)
# ===========================================================================

class HeteroGraph:
    """Undirected actor + route-context graph built from a flow frame.

    Nodes: every non-empty source-side/destination-side actor address, plus a
    route-context node per (src_chain>dst_chain|bridge).  Edges: each flow links
    its endpoint actor(s) to its context node, and (for complete flows) its two
    endpoint actors directly.  Node features are structural aggregates.
    """

    FEAT_NAMES = ["deg", "log_deg", "n_ctx", "n_chains", "mean_log_amt",
                  "mean_log_lag", "frac_complete", "frac_src", "is_ctx"]

    def __init__(self, df: pd.DataFrame):
        self.idx: Dict[str, int] = {}
        endpoints: List[Tuple[str, str, float, float, int, int]] = []
        # (actor_node, ctx_node, log_amt, log_lag, is_complete, is_src_side)
        rows: List[int] = []; cols: List[int] = []; vals: List[float] = []
        self.flow_actor_a: List[int] = []
        self.flow_actor_b: List[int] = []
        self.flow_ctx: List[int] = []

        def node(a: str) -> int:
            if a not in self.idx:
                self.idx[a] = len(self.idx)
            return self.idx[a]

        agg: Dict[int, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # per node accumulators: [deg, sum_log_amt, sum_log_lag, n_complete, n_src, _]
        node_ctx: Dict[int, set] = defaultdict(set)
        node_chains: Dict[int, set] = defaultdict(set)

        for _, r in df.iterrows():
            a_s = src_actor(r); a_d = dst_actor(r)
            ctx = _context_key(r)
            ci = node(ctx)
            amt = math.log1p(max(_to_float(r.get("src_amount")),
                                 _to_float(r.get("dst_amount")), 0.0))
            lag = math.log1p(max(_to_float(r.get("lag_seconds")), 0.0))
            is_comp = 1 if str(r.get("flow_type")) == "complete" else 0
            sc = _norm(r.get("src_chain")); dc = _norm(r.get("dst_chain"))
            ai = bi = -1
            if a_s:
                ai = node(a_s)
                rows.append(ai); cols.append(ci); vals.append(1.0)
                a = agg[ai]; a[0] += 1; a[1] += amt; a[2] += lag; a[3] += is_comp; a[4] += 1
                node_ctx[ai].add(ci)
                if sc: node_chains[ai].add(sc)
                if dc: node_chains[ai].add(dc)
            if a_d:
                bi = node(a_d)
                rows.append(bi); cols.append(ci); vals.append(1.0)
                a = agg[bi]; a[0] += 1; a[1] += amt; a[2] += lag; a[3] += is_comp
                node_ctx[bi].add(ci)
                if sc: node_chains[bi].add(sc)
                if dc: node_chains[bi].add(dc)
            if ai >= 0 and bi >= 0:
                rows.append(ai); cols.append(bi); vals.append(1.0)   # direct transfer edge
            # context node accumulators
            c = agg[ci]; c[0] += 1; c[1] += amt; c[2] += lag; c[3] += is_comp
            node_chains[ci].add(sc); node_chains[ci].add(dc)
            self.flow_actor_a.append(ai); self.flow_actor_b.append(bi); self.flow_ctx.append(ci)

        self.N = len(self.idx)
        self.ctx_ids = {i for a, i in self.idx.items() if a.startswith("ctx::")}
        # Symmetrise adjacency.
        A = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        A = A + A.T
        self.A = A
        # Node feature matrix.
        X = np.zeros((self.N, len(self.FEAT_NAMES)), dtype=np.float64)
        for i in range(self.N):
            a = agg.get(i, [0.0] * 6)
            deg = a[0]
            X[i, 0] = deg
            X[i, 1] = math.log1p(deg)
            X[i, 2] = math.log1p(len(node_ctx.get(i, ())))
            X[i, 3] = math.log1p(len(node_chains.get(i, ())))
            X[i, 4] = (a[1] / deg) if deg else 0.0
            X[i, 5] = (a[2] / deg) if deg else 0.0
            X[i, 6] = (a[3] / deg) if deg else 0.0
            X[i, 7] = (a[4] / deg) if deg else 0.0
            X[i, 8] = 1.0 if i in self.ctx_ids else 0.0
        self.X = X

    def sgc_propagate(self, K: int = 2) -> np.ndarray:
        """K-hop symmetric-normalised propagation S^K X, S = D^{-1/2}(A+I)D^{-1/2}."""
        A_hat = self.A + sp.identity(self.N, format="csr")
        deg = np.asarray(A_hat.sum(axis=1)).ravel()
        dinv = np.zeros(self.N); nz = deg > 0; dinv[nz] = 1.0 / np.sqrt(deg[nz])
        D = sp.diags(dinv)
        S = D @ A_hat @ D
        H = self.X.copy()
        for _ in range(K):
            H = S @ H
        return H

    def personalised_pagerank(self, seed_nodes: set, alpha: float = 0.15,
                              max_iter: int = 200, tol: float = 1e-12) -> np.ndarray:
        """Seeded PPR: r = (1-alpha) M r + alpha e, teleporting to the seed set."""
        if self.N == 0 or not seed_nodes:
            return np.zeros(self.N)
        deg = np.asarray(self.A.sum(axis=0)).ravel()
        dangling = deg == 0.0
        inv = np.zeros(self.N); nz = ~dangling; inv[nz] = 1.0 / deg[nz]
        M = self.A @ sp.diags(inv)
        e = np.zeros(self.N)
        for s in seed_nodes:
            e[s] = 1.0
        e /= e.sum()
        r = e.copy()
        for _ in range(max_iter):
            dmass = r[dangling].sum()
            new = (1.0 - alpha) * (M @ r + dmass * e) + alpha * e
            sm = new.sum()
            if sm > 0:
                new /= sm
            if np.abs(new - r).sum() < tol:
                r = new; break
            r = new
        return r


def learned_graph_scores(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         K: int = 2) -> Optional[Dict[str, float]]:
    """SGC-style propagation + MLP readout (transductive over train+test topology)."""
    y_train = (train_df.get(LABEL_COL, "0").astype(str) == "1").astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        return None
    n_tr = len(train_df)
    alldf = pd.concat([train_df, test_df], ignore_index=True)
    G = HeteroGraph(alldf)
    H = G.sgc_propagate(K=K)                                  # message passing

    emb_dim = H.shape[1]
    zero = np.zeros(emb_dim)

    def flow_feats(i_global: int, row) -> np.ndarray:
        ai = G.flow_actor_a[i_global]; bi = G.flow_actor_b[i_global]
        ea = H[ai] if ai >= 0 else zero
        eb = H[bi] if bi >= 0 else zero
        raw = np.array([
            math.log1p(max(_to_float(row.get("src_amount")), _to_float(row.get("dst_amount")), 0.0)),
            math.log1p(max(_to_float(row.get("lag_seconds")), 0.0)),
        ])
        return np.concatenate([ea, eb, ea * eb, raw])

    feats: List[np.ndarray] = []
    for gi, (_, row) in enumerate(alldf.iterrows()):
        feats.append(flow_feats(gi, row))
    F = np.vstack(feats)
    F_tr, F_te = F[:n_tr], F[n_tr:]

    scaler = StandardScaler().fit(F_tr)
    F_tr = scaler.transform(F_tr); F_te = scaler.transform(F_te)

    # Class-balanced bootstrap so the MLP does not collapse to the majority class.
    pos = np.where(y_train == 1)[0]; neg = np.where(y_train == 0)[0]
    neg_cap = min(len(neg), max(20000, 20 * len(pos)))
    neg_s = resample(neg, replace=False, n_samples=neg_cap, random_state=RANDOM_STATE)
    pos_s = resample(pos, replace=True, n_samples=neg_cap, random_state=RANDOM_STATE)
    sel = np.concatenate([neg_s, pos_s])
    Xb, yb = F_tr[sel], y_train[sel]

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                        alpha=1e-3, max_iter=400, random_state=RANDOM_STATE,
                        early_stopping=False)
    clf.fit(Xb, yb)
    proba = clf.predict_proba(F_te)[:, 1]
    return {test_df.iloc[k]["flow_id"]: float(proba[k]) for k in range(len(test_df))}


def cross_chain_scores(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       alpha: float = 0.15) -> Optional[Dict[str, float]]:
    """Seeded personalised-PageRank guilt-by-association over the hetero graph."""
    n_tr = len(train_df)
    alldf = pd.concat([train_df, test_df], ignore_index=True)
    G = HeteroGraph(alldf)
    # Seeds: actor nodes appearing in incident-linked TRAINING flows (label-safe).
    seeds: set = set()
    y_train = (train_df.get(LABEL_COL, "0").astype(str) == "1").astype(int).to_numpy()
    for k in range(n_tr):
        if y_train[k] == 1:
            ai = G.flow_actor_a[k]; bi = G.flow_actor_b[k]
            if ai >= 0 and ai not in G.ctx_ids:
                seeds.add(ai)
            if bi >= 0 and bi not in G.ctx_ids:
                seeds.add(bi)
    if not seeds:
        return None
    r = G.personalised_pagerank(seeds, alpha=alpha)
    out: Dict[str, float] = {}
    for k in range(len(test_df)):
        gi = n_tr + k
        ai = G.flow_actor_a[gi]; bi = G.flow_actor_b[gi]
        sa = r[ai] if ai >= 0 else 0.0
        sb = r[bi] if bi >= 0 else 0.0
        out[test_df.iloc[k]["flow_id"]] = float(sa + sb)
    return out


MODELS = {
    "pagerank": pagerank_scores,
    "learned_graph": learned_graph_scores,
    "cross_chain_specific": cross_chain_scores,
}


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _labels(df: pd.DataFrame) -> Dict[str, int]:
    return {str(r[FLOW_ID_COL]): (1 if str(r.get(LABEL_COL, "0")) == "1" else 0)
            for _, r in df.iterrows()}


def run_global(panels: Path, scores: Path, models: List[str], log) -> None:
    tr = pd.read_csv(panels / "T_train.csv", dtype=str, keep_default_na=False)
    te = pd.read_csv(panels / "T_test.csv", dtype=str, keep_default_na=False)
    log(f"[global] T_train={len(tr)} T_test={len(te)} pos_test={sum(_labels(te).values())}")
    scores.mkdir(parents=True, exist_ok=True)
    for m in models:
        res = MODELS[m](tr, te)
        if res is None:
            log(f"[global] {m}: not fittable; skipped"); continue
        out = scores / f"{m}_scores.csv"
        pd.DataFrame({"flow_id": list(res.keys()),
                      "model_score": [f"{v:.10f}" for v in res.values()]}).to_csv(out, index=False)
        met = ranked_metrics(res, _labels(te))
        log(f"[global] {m}: AP={met['average_precision']} P@10={met.get('P@10')} "
            f"P@100={met.get('P@100')} -> {out.name}")


def run_per_case(panels: Path, scores: Path, reports: Path, models: List[str], log) -> None:
    per_case = scores / "per_case"; per_case.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    for tf in sorted(panels.glob("*_train.csv")):
        case = tf.name[:-len("_train.csv")]
        if case == "T":
            continue
        te_f = panels / f"{case}_test.csv"
        if not te_f.exists():
            continue
        tr = pd.read_csv(tf, dtype=str, keep_default_na=False)
        te = pd.read_csv(te_f, dtype=str, keep_default_na=False)
        labels = _labels(te); n_pos = sum(labels.values())
        for m in models:
            res = MODELS[m](tr, te)
            if res is None:
                log(f"[per-case] {case}/{m}: not fittable (pos_test={n_pos}); skipped"); continue
            pd.DataFrame({"flow_id": list(res.keys()),
                          "model_score": [f"{v:.10f}" for v in res.values()]}
                         ).to_csv(per_case / f"{case}__{m}_scores.csv", index=False)
            met = ranked_metrics(res, labels)
            met.update({"case_id": case, "model": m, "n_train": len(tr),
                        "n_test": len(te), "n_test_pos": n_pos})
            summary.append(met)
            log(f"[per-case] {case}/{m}: AP={met['average_precision']} pos={n_pos}")
    if summary:
        cols = ["case_id", "model", "n_train", "n_test", "n_test_pos",
                "average_precision", "P@10", "R@10", "P@25", "R@25",
                "P@50", "R@50", "P@100", "R@100"]
        df = pd.DataFrame(summary)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        tag = "" if panels.name == "panels" else f"_{panels.name}"
        out = reports / f"per_case_graph_model_summary{tag}.csv"
        df[cols].to_csv(out, index=False)
        log(f"[per-case] wrote {len(summary)} rows -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--models", default="pagerank,learned_graph,cross_chain_specific")
    ap.add_argument("--regime", choices=["global", "per_case", "both"], default="both")
    ap.add_argument("--panels-subdir", default="observed_panels")
    ap.add_argument("--scores-subdir", default="scores_observed_graph")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve(); data = base / "data"
    panels = data / args.panels_subdir; scores = data / args.scores_subdir
    reports = data / "reports"; reports.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[graph_models] {msg}", flush=True)

    models = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    log(f"models={models} regime={args.regime} panels={panels.name}")
    if args.regime in ("global", "both"):
        run_global(panels, scores, models, log)
    if args.regime in ("per_case", "both"):
        run_per_case(panels, scores, reports, models, log)
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
