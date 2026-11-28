"""Early controversy prediction applied to simulated threads.

The early-controversy logistic regression is trained on the real articles outside the simulated set, with the
features a simulated thread also has: article content, the reply tree of the first K comments and their
sentiment (no timing, no users, no user graph). It then scores the 100 simulated articles from the first K
comments of each source, and the AUC is measured against the label of the real thread:
    real        the real first K comments (reference)
    A, B, C     the first K simulated comments of each condition
    retrieval   the first K real comments of the retrieval-baseline article (control: a similar real thread)
Comment labels come from the final models for real and simulated comments alike (data/simulation/generated/).
Two labels: `all` (the whole 48-hour thread, as in the main early-controversy setting) and `after_k` (only
the comments after the K-th, so the real first K comments do not share comments with the label).

    python -m src.simulation.transfer      # -> results/simulation_eval/controversy_transfer.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import controversy as cv
from ..analysis.thread_sentiment import POLARITY
from ..classifiers.common import CSV_ENCODING, run_info
from ..features.structure import tree_metrics
from ..processing.trees import ROOT, reconstructed_parents
from .evaluate import GENERATED, OUT, read_scores
from .prompts import SIM_DIR

K = cv.K
FEATURES = cv.GROUPS["content"] + ["reply_share", "max_depth", "structural_virality"] + cv.GROUPS["sentiment"]
SOURCES = ["real", "A", "B", "C", "retrieval"]


def model_labelled(c: pd.DataFrame) -> pd.DataFrame:
    """Replace the corpus labels of real comments by the final models' own labels."""
    key = c["comment_id"].astype(str)
    s = read_scores(GENERATED / "real_sentiment_scores.csv", ["id"]).set_index("id")["label"]
    u = read_scores(GENERATED / "real_incivility_scores.csv", ["id"]).set_index("id")["label"]
    c = c.assign(sentiment=key.map(s).to_numpy(), uncivil=key.map(u).astype(int).to_numpy())
    return c.assign(polarity=c["sentiment"].map(POLARITY))


def early_row(texts_parents: list[tuple], sentiment: list, uncivil: list) -> dict:
    """Structure and sentiment features of a thread's first comments given as (id, parent or ROOT)."""
    parents = dict(texts_parents)
    tree = tree_metrics(parents)
    pol = np.array([POLARITY[s] for s in sentiment])
    return {"reply_share": float(np.mean([p != ROOT for p in parents.values()])), "max_depth": tree["max_depth"],
            "structural_virality": tree["structural_virality"], "p_neg": float(np.mean(np.array(sentiment) == "neg")),
            "p_uncivil": float(np.mean(uncivil)), "polarization": float(pol.var())}


def real_early(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for aid, g in c.groupby("article_id", sort=False):
        if len(g) < K:
            continue
        e = g.iloc[:K]
        parents, _ = reconstructed_parents(e.to_dict("records"))
        rows.append({"article_id": aid, **early_row(list(parents.items()), e["sentiment"].tolist(),
                                                     e["uncivil"].tolist())})
    return pd.DataFrame(rows)


def simulated_early(run: Path) -> pd.DataFrame:
    keys = ["article_id", "condition", "id"]
    s = read_scores(run / "sentiment_scores.csv", keys).set_index(keys)["label"]
    u = read_scores(run / "incivility_scores.csv", keys).set_index(keys)["label"]
    rows = []
    with open(run / "threads.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            e = r["comments"][:K]
            k = [(r["article_id"], r["condition"], x["id"]) for x in e]
            rows.append({"article_id": r["article_id"], "source": r["condition"], **early_row(
                [(x["id"], ROOT if x["parent"] is None else x["parent"]) for x in e], [s[i] for i in k],
                [int(u[i]) for i in k])})
    return pd.DataFrame(rows)


def run(run_dir: Path) -> pd.DataFrame:
    a, c = cv.load()                                   # 48-hour comments in time order
    c = model_labelled(c)
    content = cv.early_features(a, c, K)[["article_id", *cv.GROUPS["content"]]]
    chosen = pd.read_csv(SIM_DIR / "articles.csv", dtype={"article_id": str}, encoding=CSV_ENCODING)
    retrieval = pd.read_csv(SIM_DIR / "retrieval.csv", dtype={"article_id": str, "baseline_article_id": str},
                            encoding=CSV_ENCODING)
    ids = set(chosen["article_id"])
    real = real_early(c)
    sources = [real[real["article_id"].isin(ids)].assign(source="real"), simulated_early(run_dir),
               retrieval.merge(real.rename(columns={"article_id": "baseline_article_id"}), on="baseline_article_id")
               .drop(columns=["baseline_article_id", "size", "baseline_size"]).assign(source="retrieval")]
    test = pd.concat(sources, ignore_index=True)
    rows = []
    for label_name, skip in (("all", 0), ("after_k", K)):
        labels = cv.thread_labels(c, skip)
        if skip:
            labels = labels[labels["n_48h"] >= cv.LATER_MIN]
        frame = real.merge(content, on="article_id").merge(labels, on="article_id")
        train_mask = ~frame["article_id"].isin(ids).to_numpy()
        frame["y"] = cv.make_label(frame, train_mask)    # statistics from the training articles only
        tr = frame[train_mask]
        est, grid = cv.tabular("lr")
        truth = frame.loc[~train_mask, ["article_id", "y"]]
        t = test.merge(content, on="article_id").merge(truth, on="article_id")
        _, threshold, info = cv.fit_predict(est, grid, tr[FEATURES].to_numpy(float), tr["y"].to_numpy(),
                                            t[FEATURES].to_numpy(float)[:1])
        model = info["fitted"]
        for src in SOURCES:
            d = t[t["source"] == src].dropna(subset=FEATURES)
            score = model.predict_proba(d[FEATURES].to_numpy(float))[:, 1]
            m = cv.evaluate(d["y"].to_numpy(), score, (score >= threshold).astype(int))
            rows.append({"label": label_name, "source": src, "articles": len(d), "positives": int(d["y"].sum()),
                         "train_articles": int(train_mask.sum()), "train_cv_auc": info["cv_auc"],
                         "auc": m["auc"], "auc_ci95": m["auc_ci95"], "pr_auc": m["pr_auc"], "f1": m["f1"],
                         "mean_score": round(float(score.mean()), 4)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Early controversy prediction on simulated threads")
    p.add_argument("--run", default=str(GENERATED / "main"))
    a = p.parse_args(argv)
    out = run(Path(a.run))
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "controversy_transfer.csv", index=False, encoding=CSV_ENCODING)
    (OUT / "controversy_transfer.json").write_text(json.dumps({**run_info(), "k": K, "features": FEATURES},
                                                              indent=2), encoding="utf-8")
    print(out.drop(columns=["train_articles"]).to_string(index=False))
    return 0 if not math.isnan(out["auc"].iloc[0]) else 1


if __name__ == "__main__":
    sys.exit(main())
