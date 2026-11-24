"""Sentiment model ladder on the dev set (neg / neu / pos, attitude of the comment's author).

Levels run locally: baseline (majority class, SO-PMI lexicon), classical (TF-IDF + Logistic Regression
trained on dev, scored out of fold) and pretrained ViSoBERT sentiment (5CD-AI/Vietnamese-Sentiment-visobert,
used as is). ViSoBERT fine-tuning and Qwen run on Kaggle (notebooks/sentiment_*.ipynb) and report on the
same dev items.

Data: data/tasks/sentiment/ (common.py). There is no public training set: models either do not learn
from dev (scored on all of dev) or learn from dev and are scored out of fold (5-fold StratifiedKFold,
random_state=42; hyperparameters chosen by an inner CV on the training folds only).

Ablation: the comment alone vs the comment with the comment it answers (`parent_text`) as context.

Usage (from the project root):
    python -m src.classifiers.sentiment ladder [--skip-visobert]  # local levels, writes results/sentiment/
    python -m src.classifiers.sentiment collect                   # re-score every saved dev_scores_*.csv
                                                                  # (local and kaggle/) on the current labels
    python -m src.classifiers.sentiment compare A B               # paired bootstrap of macro-F1, config A - B
    python -m src.classifiers.sentiment final CONFIG              # corpus_labels.csv from a local model
    python -m src.classifiers.sentiment final --from-file FILE    # ... from a corpus_scores file (Kaggle)

Output: results/sentiment/ (metrics.json, ladder.csv, ladder_full.csv, one dev_scores_<config>.csv per
configuration with columns id, pred, p_neg, p_neu, p_pos). Kaggle outputs use the same format and go to
results/sentiment/kaggle/. Corpus scores of a model are corpus_scores_<name>.csv in the same format, where
<name> is the dev configuration without the `_oof` suffix (the final model is refit on all of dev).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .common import (CSV_ENCODING, N_BOOT, RESULTS_DIR, SEED, dev_to_corpus_ids, load_corpus, load_dev,
                     paired_bootstrap, round_scores, run_info, task_dir, write_results)

TASK = "sentiment"
CLASSES = ["neg", "neu", "pos"]
PROBA_COLS = [f"p_{c}" for c in CLASSES]
RESULT_DIR = RESULTS_DIR / TASK
KAGGLE_RESULT_DIR = RESULT_DIR / "kaggle"
VISOBERT = "5CD-AI/Vietnamese-Sentiment-visobert"
CJK_RE = re.compile(r"[一-鿿]")

MENTION_RE = re.compile(r"@USER_[0-9a-f]+\s*:?", re.IGNORECASE)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# --------------------------------------------------------------------------- data

def load_dev_xy() -> tuple[pd.DataFrame, np.ndarray]:
    dev = load_dev(TASK)
    bad = set(dev["label"]) - set(CLASSES)
    if bad:
        raise ValueError(f"unexpected labels in dev.csv: {bad}")
    return dev, dev["label"].map(CLASSES.index).to_numpy()


def dev_labels_sha() -> str:
    """Fingerprint of the current dev labels, stored with every score so it is clear which round was used."""
    dev = load_dev(TASK)
    return hashlib.sha256("\n".join(dev["id"] + "," + dev["label"]).encode()).hexdigest()[:12]


def clean(text: str) -> str:
    """Mentions carry no sentiment; the anonymised hash would only add noise."""
    return MENTION_RE.sub(" @user ", str(text)).strip()


def find_scores(name: str) -> Path:
    """A scores file by path, or by name in results/sentiment/ then results/sentiment/kaggle/."""
    for f in (Path(name), RESULT_DIR / name, KAGGLE_RESULT_DIR / name):
        if f.is_file():
            return f
    raise FileNotFoundError(f"{name} not found in {RESULT_DIR} or {KAGGLE_RESULT_DIR}")


def read_scores(f: Path, ids) -> pd.DataFrame:
    """Scores file aligned to `ids`; raises if the item ids differ."""
    s = pd.read_csv(f, dtype={"id": str}, encoding=CSV_ENCODING)
    if len(s) != len(ids) or set(s["id"]) != set(ids):
        raise ValueError(f"{f.name}: item ids do not match")
    return s.set_index("id").loc[list(ids)].reset_index()


# --------------------------------------------------------------------------- metrics

def _macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))


def evaluate(y: np.ndarray, proba: np.ndarray, is_reply: np.ndarray | None = None) -> dict:
    pred = proba.argmax(1)
    f1s = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    rng = np.random.default_rng(SEED)
    boot = [_macro_f1(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
    out = {
        "macro_f1": round(_macro_f1(y, pred), 3),
        "macro_f1_ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
        **{f"f1_{c}": round(float(f), 3) for c, f in zip(CLASSES, f1s)},
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        **{f"pred_share_{c}": round(float((pred == k).mean()), 3) for k, c in enumerate(CLASSES)},
    }
    if is_reply is not None:
        out["macro_f1_top_level"] = round(_macro_f1(y[~is_reply], pred[~is_reply]), 3)
        out["macro_f1_reply"] = round(_macro_f1(y[is_reply], pred[is_reply]), 3)
    out["confusion"] = confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()  # rows: true neg, neu, pos
    return out


def scores_frame(ids, proba: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"id": list(ids), "pred": [CLASSES[k] for k in proba.argmax(1)]})
    for k, col in enumerate(PROBA_COLS):
        df[col] = round_scores(proba[:, k])
    return df


def proba_of(scores: pd.DataFrame) -> np.ndarray:
    return scores[PROBA_COLS].to_numpy(float)


# --------------------------------------------------------------------------- baselines

POS_SEEDS = ["tốt", "hay", "tuyệt vời", "tuyệt", "đẹp", "giỏi", "chúc mừng", "cảm ơn", "ủng hộ", "hạnh phúc",
             "vui", "xuất sắc", "đáng khen", "tự hào"]
NEG_SEEDS = ["tệ", "kém", "xấu", "buồn", "thất vọng", "bức xúc", "chán", "đau lòng", "tội nghiệp", "vô lý",
             "phẫn nộ", "lo ngại", "đáng sợ", "thương tâm"]
SOPMI_MIN_DF = 5


def normalize(text: str) -> str:
    return clean(text).lower()


def terms(text: str) -> set:
    """Lower-cased syllables and syllable bigrams (Vietnamese words are mostly one or two syllables)."""
    syl = TOKEN_RE.findall(clean(text).lower())
    return set(syl) | {f"{a} {b}" for a, b in zip(syl, syl[1:])}


def so_pmi_lexicon(texts) -> dict:
    """Semantic orientation (Turney, 2002) from co-occurrence within one comment, on unlabelled comments:
    SO(w) = log2( hits(w, POS) * hits(NEG) / (hits(w, NEG) * hits(POS)) ), with 0.01 smoothing."""
    docs = [terms(t) for t in texts]
    pos, neg = set(POS_SEEDS), set(NEG_SEEDS)
    df, with_pos, with_neg = Counter(), Counter(), Counter()
    n_pos = n_neg = 0
    for d in docs:
        hp, hn = bool(d & pos), bool(d & neg)
        n_pos += hp
        n_neg += hn
        for w in d:
            df[w] += 1
            with_pos[w] += hp
            with_neg[w] += hn
    return {w: math.log2((with_pos[w] + 0.01) * n_neg / ((with_neg[w] + 0.01) * n_pos))
            for w, c in df.items() if c >= SOPMI_MIN_DF}


def so_pmi_score(texts, lexicon: dict) -> np.ndarray:
    return np.array([sum(lexicon.get(w, 0.0) for w in terms(t)) for t in texts])


def _threshold_proba(s: np.ndarray, lo: float, hi: float, scale: float) -> np.ndarray:
    """Pseudo-probabilities whose argmax equals the threshold rule (neg < lo <= neu <= hi < pos)."""
    z = np.stack([lo - s, np.minimum(s - lo, hi - s), s - hi], 1) / scale
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def fit_thresholds(s: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 49)))
    best = (-1.0, 0.0, 0.0)
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            pred = np.where(s < lo, 0, np.where(s > hi, 2, 1))
            f = _macro_f1(y, pred)
            if f > best[0]:
                best = (f, lo, hi)
    return best[1], best[2]


def sopmi_oof(dev: pd.DataFrame, y: np.ndarray, corpus: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Lexicon from the whole unlabelled corpus; only the two thresholds learn from dev (out of fold)."""
    lexicon = so_pmi_lexicon(corpus["text"])
    s = so_pmi_score(dev["text"], lexicon)
    scale = float(np.std(s)) or 1.0
    proba = np.zeros((len(dev), 3))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(s, y):
        lo, hi = fit_thresholds(s[tr], y[tr])
        proba[te] = _threshold_proba(s[te], lo, hi, scale)
    top = sorted(lexicon.items(), key=lambda kv: kv[1])
    info = {"lexicon_size": len(lexicon), "most_negative": [w for w, _ in top[:15]],
            "most_positive": [w for w, _ in top[-15:][::-1]]}
    return proba, info


# --------------------------------------------------------------------------- classical

def _tfidf_pipeline(use_parent: bool) -> Pipeline:
    blocks = [
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b",
                                 preprocessor=normalize), "text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                 preprocessor=normalize), "text"),
    ]
    if use_parent:
        blocks.append(("parent", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                                 token_pattern=r"(?u)\b\w+\b",
                                                 preprocessor=normalize), "parent_text"))
    return Pipeline([("feat", ColumnTransformer(blocks)),
                     ("clf", LogisticRegression(max_iter=3000, random_state=SEED))])


TFIDF_GRID = {"clf__C": [0.3, 1, 3, 10], "clf__class_weight": [None, "balanced"]}


def tfidf_fit(x: pd.DataFrame, y: np.ndarray, use_parent: bool):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(_tfidf_pipeline(use_parent), TFIDF_GRID, scoring="f1_macro", cv=cv, n_jobs=-1)
    search.fit(x, y)
    return search.best_estimator_, search.best_params_


def tfidf_oof(dev: pd.DataFrame, y: np.ndarray, use_parent: bool) -> tuple[np.ndarray, list]:
    """Nested CV: the outer fold scored is never used for training or for choosing C / class weights."""
    x = dev[["text", "parent_text"]]
    proba, chosen = np.zeros((len(dev), 3)), []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y):
        model, params = tfidf_fit(x.iloc[tr], y[tr], use_parent)
        proba[te] = model.predict_proba(x.iloc[te])
        chosen.append(params)
    return proba, chosen


# --------------------------------------------------------------------------- pretrained ViSoBERT

def visobert_tokenizer():
    """transformers 5 rebuilds XLMRobertaTokenizer as a Unigram model, but ViSoBERT's tokenizer.json is BPE
    (TypeError: 'dict' object is not an instance of 'Sequence'). Loading tokenizer.json directly gives the
    same token ids as transformers 4 (checked on every dev comment and comment pair)."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(VISOBERT)
    except TypeError:
        return PreTrainedTokenizerFast.from_pretrained(VISOBERT)


def visobert_proba(texts, parents=None, batch_size: int = 32) -> np.ndarray:
    """Pretrained 5CD-AI ViSoBERT sentiment, no training. Its head is ordered NEG, POS, NEU (config.id2label);
    columns are returned in CLASSES order (neg, neu, pos). With `parents`, replies are encoded as the pair
    (comment, parent comment)."""
    import torch
    from transformers import AutoModelForSequenceClassification

    tok = visobert_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(VISOBERT).eval()
    order = [{v.lower(): k for k, v in model.config.id2label.items()}[c] for c in CLASSES]
    texts = [clean(t) for t in texts]
    parents = [clean(p) if p else None for p in parents] if parents is not None else [None] * len(texts)
    idx = np.argsort([len(t) for t in texts])  # batch similar lengths together
    out = np.zeros((len(texts), 3))
    torch.manual_seed(SEED)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            b = idx[start:start + batch_size]
            first = [texts[i] for i in b]
            second = [parents[i] for i in b]
            if any(second):
                enc = [tok(f, s, truncation=True, max_length=256) if s else tok(f, truncation=True, max_length=256)
                       for f, s in zip(first, second)]
                enc = tok.pad(enc, return_tensors="pt")
            else:
                enc = tok(first, truncation=True, max_length=256, padding=True, return_tensors="pt")
            p = torch.softmax(model(**enc).logits, -1).numpy()
            out[b] = p[:, order]
    return out


# --------------------------------------------------------------------------- ladder

def run(skip_visobert: bool = False) -> dict:
    dev, y = load_dev_xy()
    is_reply = (dev["parent_text"] != "").to_numpy()
    corpus = load_corpus(TASK)
    rows, per_config, extra = [], {}, {}

    def add(level, config, proba, train, inp, info=None):
        proba = round_scores(proba)  # score what is saved, so that ladder and collect agree
        r = {"level": level, "config": config, "train": train, "input": inp, **evaluate(y, proba, is_reply)}
        if info:
            r["info"] = info
        rows.append(r)
        per_config[f"dev_scores_{config}.csv"] = scores_frame(dev["id"], proba)
        print(f"  {config}: macro-F1 {r['macro_f1']} {r['macro_f1_ci95']}", flush=True)

    print("baselines", flush=True)
    majority = np.bincount(y, minlength=3).argmax()  # labels only count classes; no text is used
    add("baseline", "majority", np.eye(3)[np.full(len(y), majority)], "-", "text")
    proba, info = sopmi_oof(dev, y, corpus)
    extra["sopmi"] = info
    add("baseline", "sopmi_oof", proba, "corpus (unlabelled) + dev thresholds (oof)", "text", info)

    print("classical (nested CV, a few minutes)", flush=True)
    for use_parent in (False, True):
        proba, chosen = tfidf_oof(dev, y, use_parent)
        inp = "text+parent" if use_parent else "text"
        add("classical", f"tfidf_lr_{'ctx' if use_parent else 'text'}_oof", proba, "dev (oof)", inp,
            {"params_per_fold": [{k: str(v) for k, v in p.items()} for p in chosen]})

    if not skip_visobert:
        print("pretrained ViSoBERT (no training)", flush=True)
        add("deep_learning", "visobert_pre_text", visobert_proba(dev["text"]), "none (pretrained)", "text")
        add("deep_learning", "visobert_pre_ctx", visobert_proba(dev["text"], dev["parent_text"]),
            "none (pretrained)", "text+parent")

    table = pd.DataFrame(rows)
    payload = {**run_info(), "task": TASK, "eval_set": "data/tasks/sentiment/dev.csv", "dev_labels_sha": dev_labels_sha(),
               "dev_n": int(len(y)), "dev_label_counts": dict(Counter(dev["label"])), "dev_reply_share": round(float(is_reply.mean()), 3),
               "corpus_n": int(len(corpus)), "corpus_reply_share": round(float((corpus["parent_text"] != "").mean()), 3),
               "extra": extra, "results": table.to_dict(orient="records")}
    write_results(TASK, payload, {"ladder.csv": table.drop(columns=["confusion", "info"], errors="ignore"), **per_config})
    return payload


def collect() -> pd.DataFrame:
    """Score every saved dev prediction (local and Kaggle) against the *current* dev labels.
    Out-of-fold models trained on older labels must be re-run to be compared fairly; `trained_on` says so."""
    dev, y = load_dev_xy()
    is_reply = (dev["parent_text"] != "").to_numpy()
    sha = dev_labels_sha()
    local_meta = {}
    if (RESULT_DIR / "metrics.json").exists():
        m = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
        local_meta = {r["config"]: (r["level"], m.get("dev_labels_sha")) for r in m["results"]}
    rows = []
    files = sorted(RESULT_DIR.glob("dev_scores_*.csv")) + sorted(KAGGLE_RESULT_DIR.glob("dev_scores_*.csv"))
    for f in files:
        s = read_scores(f, dev["id"])
        config = f.stem[len("dev_scores_"):]
        level, trained_sha = local_meta.get(config, ("llm" if config.startswith("qwen") else "deep_learning", None))
        rows.append({"level": level, "config": config, "source": f.parent.name,
                     "trained_on": "-" if not config.endswith("oof") else (trained_sha or "see notebook"),
                     **evaluate(y, proba_of(s), is_reply)})
    full = pd.DataFrame(rows)
    full.insert(0, "dev_labels_sha", sha)
    full.to_csv(RESULT_DIR / "ladder_full.csv", index=False, encoding=CSV_ENCODING)
    return full


def cot_language_check() -> dict:
    """Chain-of-Thought outputs of the Kaggle LLM run that contain Chinese characters (the label is read
    from the digit probabilities, so it is not affected). Counts only; no text is printed."""
    out = {}
    for f in sorted(KAGGLE_RESULT_DIR.glob("*_cot_outputs.jsonl")):
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[f.name] = {"n": len(rows), "with_chinese": sum(bool(CJK_RE.search(r["output"] or "")) for r in rows)}
    return out


# --------------------------------------------------------------------------- compare

def compare(a: str, b: str) -> dict:
    """Paired bootstrap on the dev items: macro-F1 of config a minus config b, with a 95% interval."""
    dev, y = load_dev_xy()
    pred = {c: proba_of(read_scores(find_scores(f"dev_scores_{c}.csv"), dev["id"])).argmax(1) for c in (a, b)}
    return {"a": a, "b": b, "dev_labels_sha": dev_labels_sha(), "n": int(len(y)),
            "macro_f1_a": round(_macro_f1(y, pred[a]), 3), "macro_f1_b": round(_macro_f1(y, pred[b]), 3),
            "delta_macro_f1": paired_bootstrap(y, pred[a], pred[b], _macro_f1)}


# --------------------------------------------------------------------------- final labels for the corpus

LOCAL_CONFIGS = ["visobert_pre_text", "visobert_pre_ctx", "tfidf_lr_text_oof", "tfidf_lr_ctx_oof"]


def corpus_scores(config: str) -> Path:
    """Corpus scores of a local model, written to results/sentiment/corpus_scores_<name>.csv. The pretrained
    ViSoBERT is used as is; TF-IDF + LR is refit on all of dev (C and class weights chosen by CV on dev)."""
    if config not in LOCAL_CONFIGS:
        raise ValueError(f"unknown config {config}; use one of {LOCAL_CONFIGS}")
    corpus = load_corpus(TASK)
    use_parent = "_ctx" in config
    if config.startswith("visobert_pre"):
        proba = visobert_proba(corpus["text"], corpus["parent_text"] if use_parent else None)
    else:
        dev, y = load_dev_xy()
        model, _ = tfidf_fit(dev[["text", "parent_text"]], y, use_parent)
        proba = model.predict_proba(corpus[["text", "parent_text"]])
    out = RESULT_DIR / f"corpus_scores_{config.removesuffix('_oof')}.csv"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scores_frame(corpus["id"], proba).to_csv(out, index=False, encoding=CSV_ENCODING)
    return out


def final(scores_file: Path) -> dict:
    """data/tasks/sentiment/corpus_labels.csv: id, label, score. label is the model's argmax, except for dev
    items, which keep their dev label; score is the model's probability of that label. Also reports how
    often the labels agree with the argmax of every other corpus_scores_*.csv found in results/sentiment/."""
    corpus = load_corpus(TASK)
    p = proba_of(read_scores(scores_file, corpus["id"]))
    k = pd.Series(p.argmax(1), index=corpus["id"])
    dev = dev_to_corpus_ids(TASK).set_index("corpus_id")["label"].map(CLASSES.index)
    n_overridden = int((k.loc[dev.index] != dev).sum())
    k.loc[dev.index] = dev
    k = k.to_numpy()
    labels = np.array(CLASSES)[k]
    out = pd.DataFrame({"id": corpus["id"], "label": labels, "score": round_scores(p[np.arange(len(k)), k])})
    path = task_dir(TASK) / "corpus_labels.csv"
    out.to_csv(path, index=False, encoding=CSV_ENCODING)
    agreement = {}
    for f in sorted(RESULT_DIR.glob("corpus_scores_*.csv")) + sorted(KAGGLE_RESULT_DIR.glob("corpus_scores_*.csv")):
        if f.resolve() != scores_file.resolve():
            other = np.array(CLASSES)[proba_of(read_scores(f, corpus["id"])).argmax(1)]
            agreement[f.stem[len("corpus_scores_"):]] = round(float((other == labels).mean()), 3)
    return {"file": str(path), "scores_file": str(scores_file), "n": int(len(out)),
            "dev_items": int(len(load_dev(TASK))), "corpus_rows_with_dev_label": int(len(dev)),
            "dev_label_differs_from_model": n_overridden,
            "label_share": out["label"].value_counts(normalize=True).round(3).to_dict(),
            "label_agreement_with": agreement}


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sentiment classifiers on the dev set and final corpus labels")
    sub = parser.add_subparsers(dest="action")
    p = sub.add_parser("ladder", help="run the local model levels on dev")
    p.add_argument("--skip-visobert", action="store_true", help="skip the pretrained ViSoBERT (slowest level)")
    sub.add_parser("collect", help="re-score every saved dev_scores_*.csv (local and kaggle/) on the current labels")
    p = sub.add_parser("compare", help="paired bootstrap of macro-F1 between two configs (A - B)")
    p.add_argument("a", help="config name, e.g. visobert_ft_text_oof")
    p.add_argument("b", help="config name, e.g. tfidf_lr_text_oof")
    p = sub.add_parser("final", help="write data/tasks/sentiment/corpus_labels.csv")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("config", nargs="?", choices=LOCAL_CONFIGS, help="local model, scored on the corpus first")
    g.add_argument("--from-file", help="a corpus_scores_<name>.csv (e.g. from the Kaggle notebook)")
    args = parser.parse_args(argv)

    if args.action == "ladder":
        payload = run(args.skip_visobert)
        cols = ["level", "config", "input", "macro_f1", "macro_f1_ci95", "f1_neg", "f1_neu", "f1_pos",
                "macro_f1_top_level", "macro_f1_reply"]
        print("dev labels", payload["dev_labels_sha"])
        print(pd.DataFrame(payload["results"])[cols].to_string(index=False))
    elif args.action == "collect":
        cols = ["config", "source", "trained_on", "macro_f1", "macro_f1_ci95", "f1_neg", "f1_neu", "f1_pos",
                "macro_f1_top_level", "macro_f1_reply", "confusion"]
        full = collect()
        print("dev labels", full["dev_labels_sha"].iloc[0])
        print(full[cols].sort_values("macro_f1", ascending=False).to_string(index=False))
        cot = cot_language_check()
        if cot:
            print("CoT outputs with Chinese characters:", json.dumps(cot))
    elif args.action == "compare":
        print(json.dumps(compare(args.a, args.b), indent=2))
    elif args.action == "final":
        f = find_scores(args.from_file) if args.from_file else corpus_scores(args.config)
        print(json.dumps(final(f), indent=2, ensure_ascii=False))
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
