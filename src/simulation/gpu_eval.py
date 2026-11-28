"""GPU parts of the reader-simulation evaluation (run on the GPU server).

    semantic       semantic_mean_cosine of each thread: mean cosine similarity over all pairs of its comments,
                   with multilingual E5 embeddings (intfloat/multilingual-e5-base, mean pooling, "query: "
                   prefix, normalized). Writes data/simulation/generated/real_semantic.csv (every real
                   thread) and <run>/semantic.csv.
    discriminator  PhoBERT (vinai/phobert-base-v2, input segmented with pyvi as for clickbait) fine-tuned to tell
                   real from simulated comments, on the same data and 5-fold split by article as the classic
                   discriminators (evaluate.discriminator_sets); out-of-fold scores in
                   <run>/phobert_discriminator.csv (versus, model, article_id, y, score). If the PhoBERT tokenizer
                   cannot be loaded, ViSoBERT (uitnlp/visobert, no segmentation) is used and named in the file.

    python -m src.simulation.gpu_eval                 # both, on data/simulation/generated/main (about 10 minutes)
    python -m src.simulation.gpu_eval --only semantic
Then locally: python -m src.simulation.evaluate (it picks both files up).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..classifiers.common import CSV_ENCODING, SEED, run_info
from ..processing.snapshot import PROCESSED_DIR
from .evaluate import GENERATED, MENTION, discriminator_sets
from .generate import prefer_ipv4

EMBEDDER = "intfloat/multilingual-e5-base"
PHOBERT, FALLBACK = "vinai/phobert-base-v2", "uitnlp/visobert"
EPOCHS, LR, BATCH, MAX_LEN = 3, 2e-5, 32, 128


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed(texts: list[str], batch: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBEDDER)
    model = AutoModel.from_pretrained(EMBEDDER).to(device()).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok([f"query: {t}" for t in texts[i:i + batch]], padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device())
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (hidden * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def mean_pair_cosine(v: np.ndarray) -> float:
    """Mean cosine over all pairs of unit vectors: (|sum v|^2 - n) / (n (n - 1))."""
    n = len(v)
    if n < 2:
        return float("nan")
    s = v.sum(0)
    return float((s @ s - n) / (n * (n - 1)))


def semantic(run: Path) -> dict:
    c = pd.read_parquet(PROCESSED_DIR / "comments.parquet", columns=["article_id", "text"])
    c["article_id"] = c["article_id"].astype(str)
    rows = []
    with open(run / "threads.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            rows += [(r["article_id"], r["condition"], x["text"]) for x in r["comments"]]
    sim = pd.DataFrame(rows, columns=["article_id", "condition", "text"])
    texts = [MENTION.sub(" ", str(t)).strip() for t in pd.concat([c["text"], sim["text"]])]
    v = embed(texts)
    c_v, s_v = v[:len(c)], v[len(c):]
    real = pd.DataFrame([{"article_id": a, "semantic_mean_cosine": round(mean_pair_cosine(c_v[idx]), 4)}
                         for a, idx in c.groupby("article_id").indices.items()])
    real.to_csv(GENERATED / "real_semantic.csv", index=False, encoding=CSV_ENCODING)
    s = pd.DataFrame([{"article_id": a, "condition": k, "semantic_mean_cosine": round(mean_pair_cosine(s_v[idx]), 4)}
                      for (a, k), idx in sim.groupby(["article_id", "condition"]).indices.items()])
    s.to_csv(run / "semantic.csv", index=False, encoding=CSV_ENCODING)
    return {"embedder": EMBEDDER, "real_threads": len(real), "simulated_threads": len(s),
            "mean_real": round(float(real["semantic_mean_cosine"].mean()), 4),
            "mean_by_condition": s.groupby("condition")["semantic_mean_cosine"].mean().round(4).to_dict()}


def load_classifier():
    """PhoBERT with pyvi segmentation, or ViSoBERT without it when PhoBERT's tokenizer does not load."""
    from transformers import AutoTokenizer
    try:
        from pyvi import ViTokenizer
        tok = AutoTokenizer.from_pretrained(PHOBERT)
        tok("thử nghiệm")
        return PHOBERT, tok, ViTokenizer.tokenize
    except Exception as e:  # noqa: BLE001 (any loading problem means: use the fallback)
        print(f"PhoBERT unavailable ({type(e).__name__}: {e}); using {FALLBACK}", flush=True)
        return FALLBACK, AutoTokenizer.from_pretrained(FALLBACK), (lambda t: t)


def discriminator(run: Path) -> dict:
    import torch
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup

    name, tok, segment = load_classifier()
    collate = DataCollatorWithPadding(tok)
    dev = device()

    def encode(texts, labels=None):
        items = []
        for i, t in enumerate(texts):
            item = dict(tok(segment(MENTION.sub(" @user ", str(t)).strip()), truncation=True, max_length=MAX_LEN))
            if labels is not None:
                item["labels"] = int(labels[i])
            items.append(item)
        return items

    def train(texts, labels):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(dev)
        dl = DataLoader(encode(texts, labels), batch_size=BATCH, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * len(dl)
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        model.train()
        for _ in range(EPOCHS):
            for b in dl:
                model(**{k: v.to(dev) for k, v in b.items()}).loss.backward()
                opt.step()
                sch.step()
                opt.zero_grad()
        return model.eval()

    @torch.no_grad()
    def predict(model, texts):
        dl = DataLoader(encode(texts), batch_size=256, collate_fn=collate)
        return np.concatenate([torch.softmax(model(**{k: v.to(dev) for k, v in b.items()}).logits.float(), -1)
                               [:, 1].cpu().numpy() for b in dl])

    began, parts, report = time.time(), [], {"model": name, "epochs": EPOCHS, "lr": LR, "batch": BATCH,
                                             "max_len": MAX_LEN, "seed": SEED}
    for versus, d in discriminator_sets(run).items():
        texts, y, groups = d["text"].tolist(), d["y"].to_numpy(), d["article_id"].to_numpy()
        score = np.zeros(len(d))
        for fit, test in GroupKFold(5).split(texts, y, groups):
            model = train([texts[i] for i in fit], y[fit])
            score[test] = predict(model, [texts[i] for i in test])
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()
        parts.append(pd.DataFrame({"versus": versus, "model": name.split("/")[-1], "article_id": groups, "y": y,
                                   "score": np.round(score, 4)}))
        print(f"{versus} done, {time.time() - began:.0f} s", flush=True)
    pd.concat(parts).to_csv(run / "phobert_discriminator.csv", index=False, encoding=CSV_ENCODING)
    report["minutes"] = round((time.time() - began) / 60, 1)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="GPU parts of the simulation evaluation")
    p.add_argument("--run", default=str(GENERATED / "main"))
    p.add_argument("--only", choices=["semantic", "discriminator"])
    a = p.parse_args(argv)
    prefer_ipv4()
    run = Path(a.run)
    report = {**run_info(), "device": device()}
    if a.only != "discriminator":
        report["semantic"] = semantic(run)
        print(json.dumps(report["semantic"], ensure_ascii=False), flush=True)
    if a.only != "semantic":
        report["discriminator"] = discriminator(run)
    (GENERATED / "gpu_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
