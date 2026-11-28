"""Shared helpers for the three classification tasks: clickbait, sentiment, incivility.

Task data lives in data/tasks/<task>/:
    dev.csv      labelled VnExpress items; id, content columns, label. Labels were assigned by an LLM
                 annotator (Claude) following a written guide and corrected directly in this file in
                 review rounds that were also LLM-assisted (git history keeps the changes). Agreement with
                 an independent human annotator is measured on a sample (see agreement.py). There is no
                 separate test set: scores are reported on dev (models trained only on public data) or
                 with cross-validation on dev (models that also train on dev).
    corpus_labels.csv  final labels for every corpus item: id, label, score (P(label 1) for binary tasks,
                 P(assigned label) for sentiment), 4 decimals; dev items keep their dev label.
    train.csv    public training data, where one exists (clickbait: ViClickbait-2025; incivility:
                 UIT-ViCTSD, toxic comments treated as uncivil). Sentiment uses a pretrained model.
    corpus.csv   every collected article, to be labelled by the final model (clickbait only).
The comment corpus of sentiment and incivility is shared: data/tasks/comments_corpus.csv (id, text,
parent_text), every collected comment.
Build them with `python -m src.classifiers.data` after fetching the public data
(`python scripts/fetch_data.py`).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from ..processing.snapshot import PROJECT_ROOT as ROOT

SEED = 42
N_BOOT = 1000
SCORE_DECIMALS = 4
CSV_ENCODING = "utf-8-sig"  # Excel opens Vietnamese text correctly
EXTERNAL_DIR = ROOT / "data" / "external"
TASKS_DIR = ROOT / "data" / "tasks"
RESULTS_DIR = ROOT / "results"
CONTENT = {"clickbait": ["title", "sapo"], "sentiment": ["text", "parent_text"], "incivility": ["text", "parent_text"]}
LABEL_VALUES = {"clickbait": ["0", "1"], "sentiment": ["neg", "neu", "pos"], "incivility": ["0", "1"]}


def task_dir(task: str) -> Path:
    if task not in CONTENT:
        raise ValueError(f"unknown task {task}; use one of {list(CONTENT)}")
    return TASKS_DIR / task


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see src/classifiers/data.py")
    return pd.read_csv(path, dtype=str, encoding=CSV_ENCODING, keep_default_na=False)


def load_dev(task: str) -> pd.DataFrame:
    """Labelled dev set: id, content columns, label."""
    return _read(task_dir(task) / "dev.csv")


def load_train(task: str) -> pd.DataFrame:
    """Public training data: id, content columns, label (clickbait also has `source`, the news site)."""
    return _read(task_dir(task) / "train.csv")


def corpus_path(task: str) -> Path:
    """Articles for clickbait; one comment corpus shared by sentiment and incivility."""
    return task_dir(task) / "corpus.csv" if task == "clickbait" else TASKS_DIR / "comments_corpus.csv"


def load_corpus(task: str) -> pd.DataFrame:
    """Every collected item to be labelled by the final model: id and content columns."""
    return _read(corpus_path(task))


def binary_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    y, score, pred = np.asarray(y).astype(int), np.asarray(score, float), np.asarray(pred).astype(int)
    one_class = len(np.unique(y)) < 2
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "pr_auc": float("nan") if one_class else float(average_precision_score(y, score)),
        "roc_auc": float("nan") if one_class else float(roc_auc_score(y, score)),
        "predicted_positive_share": float(pred.mean()),
    }


def bootstrap_ci(y, score, pred, n: int = N_BOOT, seed: int = SEED) -> dict:
    """95% percentile intervals for F1 and PR-AUC, resampling items with replacement."""
    y, score, pred = np.asarray(y).astype(int), np.asarray(score, float), np.asarray(pred).astype(int)
    rng = np.random.default_rng(seed)
    f1s, aps = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        f1s.append(f1_score(y[idx], pred[idx], zero_division=0))
        aps.append(average_precision_score(y[idx], score[idx]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return {"f1_ci95": q(f1s), "pr_auc_ci95": q(aps)}


def evaluate_binary(y, score, pred) -> dict:
    """binary_metrics plus bootstrap CIs, rounded to 3 decimals."""
    m = binary_metrics(y, score, pred)
    m.update(bootstrap_ci(y, score, pred))
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}


def f1_threshold(y, score) -> float:
    """Threshold that maximises F1 of the positive class; ties go to the threshold closest to 0.5."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y).astype(int), np.asarray(score, float))
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)[:-1]
    best = thresholds[np.isclose(f1, f1.max(), rtol=0, atol=1e-12)]
    return float(min(best, key=lambda t: (abs(t - 0.5), t)))


def paired_bootstrap(y, pred_a, pred_b, metric, n: int = N_BOOT, seed: int = SEED) -> dict:
    """Difference metric(a) - metric(b) on the same items, with a 95% percentile interval.
    `metric(y, pred)` may take scores instead of predictions (e.g. average_precision_score)."""
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    return {"delta": round(float(metric(y, a) - metric(y, b)), 3),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 3), round(float(np.percentile(deltas, 97.5)), 3)]}


def round_scores(values, decimals: int = SCORE_DECIMALS) -> np.ndarray:
    """Every score or probability written to a CSV uses the same precision."""
    return np.round(np.asarray(values, float), decimals)


def dev_to_corpus_ids(task: str, keys: list[str] | None = None) -> pd.DataFrame:
    """Map dev items to corpus ids. dev ids are local (001, E288, H001...), so items are matched on their
    content (`keys`, default: all content columns of the task). A few corpus comments are exact
    duplicates: every copy gets the dev item's label. Returns id, corpus_id, label."""
    keys = keys or CONTENT[task]
    dev, corpus = load_dev(task), load_corpus(task)
    m = dev[["id", *keys, "label"]].merge(corpus[["id", *keys]].rename(columns={"id": "corpus_id"}),
                                          on=keys, how="left")
    missing = m.loc[m["corpus_id"].isna(), "id"]
    if len(missing):
        raise ValueError(f"{len(missing)} dev items not found in corpus.csv, e.g. dev ids {list(missing[:5])}")
    conflict = m.groupby("corpus_id")["label"].nunique()
    if (conflict > 1).any():
        raise ValueError(f"conflicting dev labels for corpus ids {list(conflict[conflict > 1].index[:5])}")
    return m[["id", "corpus_id", "label"]].reset_index(drop=True)


def hardware() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version()}


def run_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT, check=False).stdout.strip() or None
    except OSError:
        commit = None
    return {"data": "data/tasks", "seed": SEED, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit}


def write_results(name: str, payload: dict, tables: dict[str, pd.DataFrame]) -> Path:
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for fname, df in tables.items():
        df.to_csv(out / fname, index=False, encoding=CSV_ENCODING)
    return out
