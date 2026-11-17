"""Build train.csv and corpus.csv of the three tasks in data/tasks/<task>/ (see common.py).

    clickbait   train: ViClickbait-2025 (id, title, sapo, source, label 0/1)
                corpus: every article (id = article_id, title, sapo)
    incivility  train: UIT-ViCTSD, all 10,000 comments (id, text, parent_text empty, label = Toxicity).
                Toxic comments are treated as uncivil; the original split is kept in the id.
                corpus: every comment (id = comment_id, text, parent_text)
    sentiment   no public training set (a pretrained ViSoBERT sentiment model is used)
                corpus: same comments as incivility

dev.csv files are not built here: they are the labelled VnExpress items, corrected by hand in place.
Public data must be fetched first (`python scripts/fetch_data.py`).

Usage (from the project root):
    python -m src.classifiers.data
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ..processing.snapshot import PROCESSED_DIR
from ..processing.trees import ROOT as TREE_ROOT, reconstructed_parents
from .common import CSV_ENCODING, EXTERNAL_DIR, task_dir


def clickbait_train() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_DIR / "viclickbait2025" / "clickbait_dataset_vietnamese.csv")
    return pd.DataFrame({"id": df["id"], "title": df["title"].fillna(""), "sapo": df["lead_paragraph"].fillna(""),
                         "source": df["source"], "label": (df["label"] == "clickbait").astype(int)})


def incivility_train() -> pd.DataFrame:
    parts = []
    for split in ("train", "valid", "test"):
        df = pd.read_csv(EXTERNAL_DIR / "victsd" / f"ViCTSD_{split}.csv")
        parts.append(pd.DataFrame({"id": [f"victsd_{split}_{i}" for i in range(len(df))],
                                   "text": df["Comment"].fillna("").astype(str), "parent_text": "",
                                   "label": df["Toxicity"].astype(int)}))
    return pd.concat(parts, ignore_index=True)


def article_corpus() -> pd.DataFrame:
    a = pd.read_parquet(PROCESSED_DIR / "articles.parquet", columns=["article_id", "title", "sapo"])
    return pd.DataFrame({"id": a["article_id"].astype(str), "title": a["title"].fillna(""), "sapo": a["sapo"].fillna("")})


def comment_corpus() -> pd.DataFrame:
    """All comments with the text of the comment they answer (recovered reply tree; empty for top level)."""
    comments = pd.read_parquet(PROCESSED_DIR / "comments.parquet")
    text = comments.set_index("comment_id")["text"]
    parent_text = {}
    for _, g in comments.groupby("article_id"):
        parents, _ = reconstructed_parents(g.to_dict("records"))
        for cid, p in parents.items():
            parent_text[cid] = "" if p == TREE_ROOT else text[p]
    return pd.DataFrame({"id": comments["comment_id"].astype(str), "text": comments["text"].fillna(""),
                         "parent_text": comments["comment_id"].map(parent_text).fillna("")})


def build() -> dict:
    out = {}
    files = {("clickbait", "train"): clickbait_train(), ("incivility", "train"): incivility_train(),
             ("clickbait", "corpus"): article_corpus()}
    comments = comment_corpus()
    files[("sentiment", "corpus")] = comments
    files[("incivility", "corpus")] = comments
    for (task, name), df in files.items():
        d = task_dir(task)
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f"{name}.csv", index=False, encoding=CSV_ENCODING)
        out[f"{task}/{name}.csv"] = int(len(df))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build train.csv and corpus.csv for the three tasks")
    a = p.parse_args(argv)
    for k, v in build().items():
        print(f"{k}: {v} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
