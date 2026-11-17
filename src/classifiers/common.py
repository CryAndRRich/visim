"""Shared helpers for the three classification tasks: clickbait, sentiment, incivility.

Task data lives in data/tasks/<task>/:
    dev.csv      labelled VnExpress items; id, content columns, label. Labels were first assigned by an LLM
                 annotator (Claude) following a written guide, then corrected by hand directly in this
                 file (git history keeps the changes). There is no separate test set: scores are reported
                 on dev (models trained only on public data) or with cross-validation on dev (models that
                 also train on dev).
    train.csv    public training data, where one exists (clickbait: ViClickbait-2025; incivility:
                 UIT-ViCTSD, toxic comments treated as uncivil). Sentiment uses a pretrained model.
    corpus.csv   every collected item, to be labelled by the final model (articles for clickbait,
                 comments for sentiment and incivility).
Build them with `python -m src.classifiers.data` after fetching the public data
(`python scripts/fetch_data.py`).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from ..processing.snapshot import PROJECT_ROOT as ROOT

SEED = 42
N_BOOT = 1000
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


def load_corpus(task: str) -> pd.DataFrame:
    """Every collected item to be labelled by the final model: id and content columns."""
    return _read(task_dir(task) / "corpus.csv")


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
