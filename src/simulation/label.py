"""Sentiment and incivility labels for simulated comments, from the same final models as the real corpus.

    incivility   the final TF-IDF + logistic regression model (trained on dev, comment text only, F1 threshold),
                 refitted here (deterministic, CPU, under a minute). Checks that the refit reproduces the corpus
                 labels on non-dev comments.
    sentiment    the final ViSoBERT model (5CD-AI/Vietnamese-Sentiment-visobert fine-tuned on dev, comment text
                 only, 4 epochs, lr 2e-5, batch 16, max length 256, mean probabilities of seeds 42, 43, 44),
                 retrained here because its weights were not kept (GPU, a few minutes). It scores the simulated
                 comments and every real comment, so real and simulated threads are compared with one model;
                 the agreement with the corpus labels on non-dev comments measures how close the retrained model
                 is to the original one.

Output, next to each run's threads.jsonl: <task>_scores.csv (article_id, condition, id, label, score...). The real
comments' sentiment scores of the retrained model go to data/simulation/generated/real_sentiment_scores.csv.

    python -m src.simulation.label incivility                         # local
    python -m src.simulation.label sentiment                          # GPU server
    python -m src.simulation.label sentiment --runs data/simulation/generated/main data/simulation/generated/rewrite
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..classifiers.common import CSV_ENCODING, dev_to_corpus_ids, load_corpus, load_dev, run_info
from .prompts import SIM_DIR

GENERATED = SIM_DIR / "generated"
DEFAULT_RUNS = [str(GENERATED / "main")]
CLASSES = ["neg", "neu", "pos"]
SENTIMENT_MODEL = "5CD-AI/Vietnamese-Sentiment-visobert"
SEEDS = [42, 43, 44]
EPOCHS, LR, BATCH, MAX_LEN = 4, 2e-5, 16, 256
INCIVILITY_CONFIG = "tfidf_lr_dev_comment_tf1"
MENTION = re.compile(r"@USER_[0-9a-f]+\s*:?", re.IGNORECASE)


def simulated(run: str) -> pd.DataFrame:
    rows = []
    with open(Path(run) / "threads.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            rows += [{"article_id": r["article_id"], "condition": r["condition"], "id": c["id"], "text": c["text"]}
                     for c in r["comments"]]
    return pd.DataFrame(rows)


def non_dev_agreement(task: str, ids: pd.Series, labels: np.ndarray) -> dict:
    """Agreement of new labels with corpus_labels.csv on comments outside dev (dev rows keep their dev label)."""
    old = pd.read_csv(SIM_DIR.parent / "tasks" / task / "corpus_labels.csv", dtype=str, encoding=CSV_ENCODING)
    dev_ids = set(dev_to_corpus_ids(task)["corpus_id"].astype(str))
    new = pd.DataFrame({"id": ids.astype(str).to_numpy(), "new": np.asarray(labels).astype(str)})
    m = new.merge(old, on="id")
    m = m[~m["id"].isin(dev_ids)]
    return {"non_dev_comments": len(m), "agreement": round(float((m["new"] == m["label"]).mean()), 4)}


def incivility(runs: list[str]) -> dict:
    from ..classifiers import incivility as inc

    dev = load_dev("incivility")
    train, _ = inc.public_training(dev)
    fitted = inc.fit_tfidf(train, dev, "dev", False)
    threshold = fitted[2]["threshold_f1"]
    corpus = load_corpus("incivility")
    report = {"config": INCIVILITY_CONFIG, **fitted[2],
              "reproduction": non_dev_agreement(
                  "incivility", corpus["id"], (inc.tfidf_scores(fitted, corpus, False) >= threshold).astype(int))}
    for run in runs:
        sim = simulated(run)
        score = inc.tfidf_scores(fitted, sim.assign(parent_text=""), False)
        out = sim.drop(columns="text").assign(label=(score >= threshold).astype(int), score=np.round(score, 4))
        out.to_csv(Path(run) / "incivility_scores.csv", index=False, encoding=CSV_ENCODING)
        report[Path(run).name] = {"comments": len(out), "uncivil_share": round(float(out["label"].mean()), 4)}
    return report


def sentiment(runs: list[str]) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              PreTrainedTokenizerFast, get_linear_schedule_with_warmup)

    from .generate import prefer_ipv4

    prefer_ipv4()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    except TypeError:   # transformers 5 reads ViSoBERT's BPE tokenizer.json only this way
        tok = PreTrainedTokenizerFast.from_pretrained(SENTIMENT_MODEL)
    id2label = AutoConfig.from_pretrained(SENTIMENT_MODEL).id2label          # not in our class order
    head = [{v.lower(): k for k, v in id2label.items()}[c] for c in CLASSES]
    collate = DataCollatorWithPadding(tok)

    def clean(t):
        return MENTION.sub(" @user ", str(t)).strip()

    def encode(texts, labels=None):
        items = []
        for i, t in enumerate(texts):
            item = dict(tok(clean(t), truncation=True, max_length=MAX_LEN))
            if labels is not None:
                item["labels"] = head[int(labels[i])]
            items.append(item)
        return items

    def train(texts, labels, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).to(device)
        dl = DataLoader(encode(texts, labels), batch_size=BATCH, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * len(dl)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        model.train()
        for _ in range(EPOCHS):
            for b in dl:
                model(**{k: v.to(device) for k, v in b.items()}).loss.backward()
                opt.step()
                sch.step()
                opt.zero_grad()
        return model.eval()

    @torch.no_grad()
    def predict(model, texts):
        dl = DataLoader(encode(texts), batch_size=128, collate_fn=collate)
        probs = [torch.softmax(model(**{k: v.to(device) for k, v in b.items()}).logits.float(), -1).cpu().numpy()
                 for b in dl]
        return np.concatenate(probs)[:, head]

    began = time.time()
    dev = load_dev("sentiment")
    y = dev["label"].map(CLASSES.index).to_numpy()
    corpus = load_corpus("sentiment")
    sims = {run: simulated(run) for run in runs}
    real_p, sim_p = [], {run: [] for run in runs}
    for seed in SEEDS:
        model = train(dev["text"].tolist(), y, seed)
        real_p.append(predict(model, corpus["text"].tolist()))
        for run, sim in sims.items():
            sim_p[run].append(predict(model, sim["text"].tolist()))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"seed {seed} done, {time.time() - began:.0f} s", flush=True)

    def table(frame, probs):
        p = np.mean(probs, 0)
        out = frame.assign(label=[CLASSES[k] for k in p.argmax(1)])
        for k, c in enumerate(CLASSES):
            out[f"p_{c}"] = p[:, k].round(4)
        return out

    real = table(corpus[["id"]], real_p)
    real.to_csv(GENERATED / "real_sentiment_scores.csv", index=False, encoding=CSV_ENCODING)
    report = {"model": SENTIMENT_MODEL, "seeds": SEEDS, "epochs": EPOCHS, "lr": LR, "batch": BATCH,
              "max_len": MAX_LEN, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "reproduction": non_dev_agreement("sentiment", real["id"], real["label"].to_numpy())}
    for run, sim in sims.items():
        out = table(sim.drop(columns="text"), sim_p[run])
        out.to_csv(Path(run) / "sentiment_scores.csv", index=False, encoding=CSV_ENCODING)
        report[Path(run).name] = {"comments": len(out), **out["label"].value_counts(normalize=True).round(4).to_dict()}
    report["minutes"] = round((time.time() - began) / 60, 1)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Label simulated comments with the final classifiers")
    p.add_argument("task", choices=["incivility", "sentiment"])
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run folders holding threads.jsonl")
    a = p.parse_args(argv)
    report = {**run_info(), **(incivility(a.runs) if a.task == "incivility" else sentiment(a.runs))}
    (GENERATED / f"{a.task}_labels.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
