#!/usr/bin/env python3
"""Stage 6 (ML): Per-case + global baseline model fitting and ranked scoring.

This module fits the supervised/unsupervised tabular baselines used for
post-incident cross-chain flow prioritisation, under the panel-wise
chronological split produced by Stage 4-5 (build_panels_and_split.py).

Models
------
    logistic_regression  : LogisticRegression(class_weight="balanced")
    random_forest        : RandomForestClassifier(class_weight="balanced")
    gradient_boosting    : XGBoost (if importable) else sklearn
                           HistGradientBoostingClassifier
    isolation_forest     : IsolationForest (unsupervised anomaly score)

Two evaluation regimes are produced:

  * GLOBAL  -- trained on data/panels/T_train.csv, scored on
               data/panels/T_test.csv.  Scores written to
               data/scores/<model>_scores.csv so the existing Stage 7
               ranked_evaluation.py consumes them unchanged.

  * PER-CASE -- for every <case>_train.csv / <case>_test.csv pair, a
               separate model is fitted on that case's training partition
               and scored on its own test partition (the paper's
               panel-wise design).  Scores written to
               data/scores/per_case/<case>__<model>_scores.csv and a
               per-case metrics roll-up to
               data/reports/per_case_model_summary.csv.

Imbalance handling
------------------
Supervised models use class_weight="balanced" (no synthetic oversampling,
so every training row remains a real observed flow).  IsolationForest is
unsupervised and ignores labels at fit time.

Leakage control
---------------
Behavioural frequency features (sender / recipient occurrence counts) are
computed ONLY on the training partition and then applied to the test
partition.  Categorical vocabularies are likewise learnt on train; unseen
categories at test time map to the OneHotEncoder "infrequent/unknown"
bucket.  The chronological split itself is performed upstream in Stage 4-5.

NOTE: This script is intended to be run with the project ML virtualenv:
    venv_models/bin/python3 train_models_per_case.py
It is deliberately NOT invoked by run_overnight.sh; model training is a
separate, explicit step (per project instruction, models are prepared but
not trained automatically).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    IsolationForest,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional XGBoost (preferred gradient-boosting impl when its native libs
# are available; otherwise we fall back to sklearn HistGradientBoosting).
try:
    from xgboost import XGBClassifier  # type: ignore
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False


RANDOM_STATE = 42
LABEL_COL = "incident_linked"
FLOW_ID_COL = "flow_id"

# Raw matched-flow columns used to derive features.
CATEGORICAL_COLS = ["affected_bridge_protocol", "src_chain", "dst_chain", "route"]
NUMERIC_COLS = [
    "log_src_amount",
    "log_dst_amount",
    "log_lag_seconds",
    "amount_ratio",
    "sender_recipient_equal",
    "src_sender_freq",
    "dst_recipient_freq",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _to_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


def _norm_addr(x: Any) -> str:
    s = "" if x is None else str(x).strip().lower()
    return s


def build_frequency_maps(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Training-derived sender / recipient occurrence counts (leakage-safe)."""
    sender_freq = Counter(_norm_addr(a) for a in train_df.get("src_sender", []))
    recip_freq = Counter(_norm_addr(a) for a in train_df.get("dst_recipient", []))
    sender_freq.pop("", None)
    recip_freq.pop("", None)
    return dict(sender_freq), dict(recip_freq)


def featurise(df: pd.DataFrame,
              sender_freq: Dict[str, int],
              recip_freq: Dict[str, int]) -> pd.DataFrame:
    """Map raw matched-flow rows to the model feature frame.

    `sender_freq` / `recip_freq` are computed on the TRAIN partition and
    passed in so the same mapping is applied to test rows (no leakage)."""
    out = pd.DataFrame(index=df.index)

    # Categorical
    out["affected_bridge_protocol"] = df.get("affected_bridge_protocol", "").fillna("").astype(str)
    out["src_chain"] = df.get("src_chain", "").fillna("").astype(str)
    out["dst_chain"] = df.get("dst_chain", "").fillna("").astype(str)
    out["route"] = out["src_chain"] + "->" + out["dst_chain"]

    # Numeric (log1p-compressed amounts + lag)
    src_amt = df.get("src_amount", pd.Series(index=df.index)).map(_to_float)
    dst_amt = df.get("dst_amount", pd.Series(index=df.index)).map(_to_float)
    lag = df.get("lag_seconds", pd.Series(index=df.index)).map(_to_float)
    out["log_src_amount"] = np.log1p(src_amt.clip(lower=0))
    out["log_dst_amount"] = np.log1p(dst_amt.clip(lower=0))
    out["log_lag_seconds"] = np.log1p(lag.clip(lower=0))
    # Amount ratio (dst/src), bounded; 0 when src missing.
    denom = src_amt.replace(0, np.nan)
    out["amount_ratio"] = (dst_amt / denom).fillna(0.0).clip(lower=0, upper=10.0)

    # Sender == recipient indicator
    s = df.get("src_sender", pd.Series(index=df.index)).map(_norm_addr)
    r = df.get("dst_recipient", pd.Series(index=df.index)).map(_norm_addr)
    out["sender_recipient_equal"] = ((s != "") & (s == r)).astype(float)

    # Training-derived frequency features
    out["src_sender_freq"] = s.map(lambda a: float(sender_freq.get(a, 0)))
    out["dst_recipient_freq"] = r.map(lambda a: float(recip_freq.get(a, 0)))

    return out


def _build_preprocessor() -> ColumnTransformer:
    """OneHotEncoder for categoricals (unknown-safe) + StandardScaler for numerics."""
    # handle_unknown="infrequent_if_exist" keeps unseen test categories from erroring.
    try:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=1, sparse_output=False)
    except TypeError:
        # Older sklearn: fall back to ignore + dense.
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("cat", ohe, CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(name: str):
    """Return an (estimator, is_supervised) pair for the given model name."""
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE), True
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE), True
    if name == "gradient_boosting":
        if _HAVE_XGB:
            return XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=RANDOM_STATE,
                n_jobs=-1), True
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05,
            random_state=RANDOM_STATE), True
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE), True
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=600,
            early_stopping=True, random_state=RANDOM_STATE), True
    if name == "isolation_forest":
        return IsolationForest(
            n_estimators=300, contamination="auto",
            random_state=RANDOM_STATE, n_jobs=-1), False
    raise ValueError(f"unknown model name: {name!r}")


# ---------------------------------------------------------------------------
# Train / score one (train_df, test_df) pair for one model
# ---------------------------------------------------------------------------

def fit_score(model_name: str,
              train_df: pd.DataFrame,
              test_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Fit `model_name` on train_df, score test_df.

    Returns a DataFrame [flow_id, model_score] or None when the panel
    cannot support the model (e.g. supervised model with a single class
    in the training partition)."""
    if test_df.empty:
        return None

    sender_freq, recip_freq = build_frequency_maps(train_df)
    X_train = featurise(train_df, sender_freq, recip_freq)
    X_test = featurise(test_df, sender_freq, recip_freq)

    estimator, supervised = make_model(model_name)
    pre = _build_preprocessor()

    if supervised:
        y_train = (train_df.get(LABEL_COL, "0").astype(str) == "1").astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            # Cannot fit a supervised classifier on a single-class panel.
            return None
        pipe = Pipeline([("pre", pre), ("clf", estimator)])
        pipe.fit(X_train, y_train)
        # Probability of the positive (incident-linked) class.
        proba = pipe.predict_proba(X_test)[:, 1]
        scores = proba
    else:
        # Unsupervised: fit on train features (labels ignored), score test.
        pipe = Pipeline([("pre", pre), ("clf", estimator)])
        pipe.fit(X_train)
        # IsolationForest: higher score_samples == more normal; negate so
        # that larger model_score == more anomalous == higher review priority.
        raw = pipe.named_steps["clf"].score_samples(
            pipe.named_steps["pre"].transform(X_test))
        scores = -raw

    return pd.DataFrame({
        FLOW_ID_COL: test_df.get(FLOW_ID_COL, pd.Series(index=test_df.index)).astype(str).to_numpy(),
        "model_score": scores,
    })


# ---------------------------------------------------------------------------
# Ranked metrics (self-contained so per-case eval needs no Stage 7)
# ---------------------------------------------------------------------------

def ranked_metrics(scores_df: pd.DataFrame,
                   labels: Dict[str, int],
                   k_cutoffs=(10, 25, 50, 100)) -> Dict[str, Any]:
    pairs = []
    for _, row in scores_df.iterrows():
        fid = str(row[FLOW_ID_COL])
        pairs.append((float(row["model_score"]), labels.get(fid, 0), fid))
    pairs.sort(key=lambda t: (-t[0], t[2]))
    n = len(pairs)
    total_pos = sum(y for _, y, _ in pairs)
    out: Dict[str, Any] = {"n_test": n, "n_test_pos": total_pos}
    cum = 0
    ap = 0.0
    prev_r = 0.0
    for i, (_s, y, _f) in enumerate(pairs, start=1):
        cum += y
        if i in k_cutoffs:
            out[f"P@{i}"] = round(cum / i, 6)
            out[f"R@{i}"] = round((cum / total_pos) if total_pos else 0.0, 6)
        if total_pos:
            p_i = cum / i
            r_i = cum / total_pos
            ap += (r_i - prev_r) * p_i
            prev_r = r_i
    out["average_precision"] = round(ap, 6)
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _labels_from(df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _, row in df.iterrows():
        out[str(row.get(FLOW_ID_COL, ""))] = 1 if str(row.get(LABEL_COL, "0")) == "1" else 0
    return out


def run_global(panels_dir: Path, scores_dir: Path, models: List[str], log) -> None:
    t_train_p = panels_dir / "T_train.csv"
    t_test_p = panels_dir / "T_test.csv"
    if not (t_train_p.exists() and t_test_p.exists()):
        log(f"[global] missing T_train/T_test; skipping global regime")
        return
    train_df = pd.read_csv(t_train_p, dtype=str, keep_default_na=False)
    test_df = pd.read_csv(t_test_p, dtype=str, keep_default_na=False)
    log(f"[global] T_train={len(train_df)} T_test={len(test_df)}")
    scores_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        res = fit_score(m, train_df, test_df)
        if res is None:
            log(f"[global] {m}: panel not fittable (single-class or empty); skipped")
            continue
        out = scores_dir / f"{m}_scores.csv"
        res.to_csv(out, index=False)
        log(f"[global] {m}: wrote {len(res)} scores -> {out.name}")


def run_per_case(panels_dir: Path, scores_dir: Path, reports_dir: Path,
                 models: List[str], log) -> None:
    per_case_dir = scores_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    train_files = sorted(panels_dir.glob("*_train.csv"))
    for tf in train_files:
        case_id = tf.name[:-len("_train.csv")]
        if case_id in ("T", ):  # skip the pooled T_train if it ever matches
            continue
        test_f = panels_dir / f"{case_id}_test.csv"
        if not test_f.exists():
            continue
        train_df = pd.read_csv(tf, dtype=str, keep_default_na=False)
        test_df = pd.read_csv(test_f, dtype=str, keep_default_na=False)
        labels = _labels_from(test_df)
        n_pos_test = sum(labels.values())
        for m in models:
            res = fit_score(m, train_df, test_df)
            if res is None:
                log(f"[per-case] {case_id} / {m}: not fittable "
                    f"(train={len(train_df)} test={len(test_df)} pos_test={n_pos_test}); skipped")
                continue
            out = per_case_dir / f"{case_id}__{m}_scores.csv"
            res.to_csv(out, index=False)
            met = ranked_metrics(res, labels)
            met.update({"case_id": case_id, "model": m,
                        "n_train": len(train_df), "n_test": len(test_df),
                        "n_test_pos": n_pos_test})
            summary.append(met)
            log(f"[per-case] {case_id} / {m}: AP={met.get('average_precision')} "
                f"test={len(test_df)} pos={n_pos_test} -> {out.name}")
    if summary:
        cols = ["case_id", "model", "n_train", "n_test", "n_test_pos",
                "average_precision", "P@10", "R@10", "P@25", "R@25",
                "P@50", "R@50", "P@100", "R@100"]
        df = pd.DataFrame(summary)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
        tag = "" if panels_dir.name == "panels" else f"_{panels_dir.name}"
        out = reports_dir / f"per_case_model_summary{tag}.csv"
        df.to_csv(out, index=False)
        log(f"[per-case] wrote summary ({len(summary)} rows) -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--models",
                    default="logistic_regression,random_forest,gradient_boosting,isolation_forest")
    ap.add_argument("--regime", choices=["global", "per_case", "both"], default="both")
    ap.add_argument("--panels-subdir", default="panels",
                    help="Panel directory under data/ (e.g. 'panels' for matched "
                         "complete flows, 'observed_panels' for the matched+single-side set).")
    ap.add_argument("--scores-subdir", default="scores",
                    help="Scores output directory under data/ (use a distinct dir, "
                         "e.g. 'scores_observed', to keep benchmarks separate).")
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    data = base / "data"
    panels_dir = data / args.panels_subdir
    scores_dir = data / args.scores_subdir
    reports_dir = data / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[train_models_per_case] {msg}", flush=True)

    log(f"xgboost_available={_HAVE_XGB} (else HistGradientBoosting)")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    log(f"models={models} regime={args.regime}")

    if args.regime in ("global", "both"):
        run_global(panels_dir, scores_dir, models, log)
    if args.regime in ("per_case", "both"):
        run_per_case(panels_dir, scores_dir, reports_dir, models, log)

    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
