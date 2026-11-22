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
    python -m src.classifiers.sentiment ladder                 # local levels, writes results/sentiment/
    python -m src.classifiers.sentiment collect                # re-score every saved dev_scores_*.csv
                                                               # (local and kaggle/) on the current labels
    python -m src.classifiers.sentiment review                 # review sheet for the next labelling round
    python -m src.classifiers.sentiment label N                # label round N in the terminal (resumable)
    python -m src.classifiers.sentiment apply-review N         # compare round N decisions with dev labels
    python -m src.classifiers.sentiment apply-review N --write # ... and write them into dev.csv
    python -m src.classifiers.sentiment predict-corpus CONFIG  # corpus scores of a local model
    python -m src.classifiers.sentiment label-corpus FILE      # corpus_labels.csv from a corpus scores file

Output: results/sentiment/ (metrics.json, ladder.csv, ladder_full.csv, one dev_scores_<config>.csv per
configuration with columns id, pred, p_neg, p_neu, p_pos). Kaggle outputs use the same format and go to
results/sentiment/kaggle/. Review sheets go to notes/sentiment_review/ (committed: they record which items
were checked in which round).
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

from .common import CSV_ENCODING, RESULTS_DIR, ROOT, SEED, load_corpus, load_dev, run_info, task_dir, write_results

TASK = "sentiment"
CLASSES = ["neg", "neu", "pos"]
PROBA_COLS = [f"p_{c}" for c in CLASSES]
RESULT_DIR = RESULTS_DIR / TASK
KAGGLE_RESULT_DIR = RESULT_DIR / "kaggle"
REVIEW_DIR = ROOT / "notes" / "sentiment_review"
VISOBERT = "5CD-AI/Vietnamese-Sentiment-visobert"
N_BOOT = 1000
N_PER_GROUP = 10

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


def dev_to_corpus_ids() -> pd.DataFrame:
    """dev ids are local (001..); map them to corpus comment ids by (text, parent_text).
    A few comments are exact duplicates in the corpus: every copy gets the dev item's label."""
    dev, corpus = load_dev(TASK), load_corpus(TASK)
    m = dev[["id", "text", "parent_text", "label"]].merge(corpus.rename(columns={"id": "corpus_id"}),
                                                           on=["text", "parent_text"], how="left")
    missing = m["corpus_id"].isna().sum()
    if missing:
        raise ValueError(f"{missing} dev items not found in corpus.csv")
    return m[["id", "corpus_id", "label"]]


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
        df[col] = np.round(proba[:, k], 4)
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

def visobert_proba(texts, parents=None, batch_size: int = 32) -> np.ndarray:
    """Pretrained 5CD-AI ViSoBERT sentiment, no training. Its head is ordered NEG, POS, NEU (config.id2label);
    columns are returned in CLASSES order (neg, neu, pos). With `parents`, replies are encoded as the pair
    (comment, parent comment)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(VISOBERT)
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
        s = pd.read_csv(f, dtype={"id": str}, encoding=CSV_ENCODING)
        if set(s["id"]) != set(dev["id"]):
            raise ValueError(f"{f.name}: item ids do not match the dev set")
        s = s.set_index("id").loc[dev["id"]]
        config = f.stem[len("dev_scores_"):]
        level, trained_sha = local_meta.get(config, ("llm" if config.startswith("qwen") else "deep_learning", None))
        rows.append({"level": level, "config": config, "source": f.parent.name,
                     "trained_on": "-" if not config.endswith("oof") else (trained_sha or "see notebook"),
                     **evaluate(y, proba_of(s), is_reply)})
    full = pd.DataFrame(rows).drop(columns=["confusion"])
    full.insert(0, "dev_labels_sha", sha)
    full.to_csv(RESULT_DIR / "ladder_full.csv", index=False, encoding=CSV_ENCODING)
    return full


# --------------------------------------------------------------------------- review rounds

def _checked_ids() -> set:
    return {i for f in REVIEW_DIR.glob("round_*_key.csv") for i in pd.read_csv(f, dtype=str)["id"]}


def _next_round() -> int:
    done = [int(re.search(r"round_(\d+)_key", f.name).group(1)) for f in REVIEW_DIR.glob("round_*_key.csv")]
    return max(done, default=0) + 1


def review(scores_file: Path, n: int = N_PER_GROUP) -> Path:
    """TASK.md section 4: for each class the n items the model is most sure belong to it, plus the n items
    with the smallest gap between the two most likely classes; items checked in earlier rounds are skipped.
    The sheet hides the current label and the model's guess so the reviewer decides first."""
    dev, _ = load_dev_xy()
    s = pd.read_csv(scores_file, dtype={"id": str}, encoding=CSV_ENCODING).set_index("id")
    s = s.drop(index=list(_checked_ids() & set(s.index)))
    p = s[PROBA_COLS]
    picked, groups = [], {}
    for c, col in zip(CLASSES, PROBA_COLS):
        for i in p[~p.index.isin(picked)][col].nlargest(n).index:
            picked.append(i)
            groups[i] = f"sure_{c}"
    top2 = np.sort(p.to_numpy(), 1)[:, -2:]
    margin = pd.Series(top2[:, 1] - top2[:, 0], index=p.index)
    for i in margin[~margin.index.isin(picked)].nsmallest(n).index:
        picked.append(i)
        groups[i] = "uncertain"

    rnd = _next_round()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = dev.set_index("id").loc[picked]
    order = np.random.default_rng(SEED + rnd).permutation(len(picked))
    sheet = pd.DataFrame({"id": np.array(picked)[order], "text": rows["text"].to_numpy()[order],
                          "parent_text": rows["parent_text"].to_numpy()[order], "my_label": "",
                          "labeled_by": "", "reason": ""})
    key = pd.DataFrame({"id": picked, "group": [groups[i] for i in picked], "model_pred": s.loc[picked, "pred"].to_numpy(),
                        **{c: s.loc[picked, c].to_numpy() for c in PROBA_COLS},
                        "label_before": rows["label"].to_numpy(), "scores_file": scores_file.name})
    sheet_path = REVIEW_DIR / f"round_{rnd}.csv"
    sheet.to_csv(sheet_path, index=False, encoding=CSV_ENCODING)
    key.to_csv(REVIEW_DIR / f"round_{rnd}_key.csv", index=False, encoding=CSV_ENCODING)
    return sheet_path


LABEL_KEYS = {"1": "neg", "2": "neu", "3": "pos", "n": "neg", "u": "neu", "p": "pos"}
GUIDE = """Gán theo thái độ của NGƯỜI VIẾT (TASK.md mục 6), không theo tin vui hay tin buồn.
  1 / n = neg: chỉ trích, bức xúc, mỉa mai, lo lắng, buồn, chia buồn
  2 / u = neu: hỏi thông tin, nêu sự kiện, không rõ thái độ
  3 / p = pos: khen, ủng hộ, vui mừng, chúc mừng
  Mỉa mai bằng từ tích cực vẫn là neg. Đồng tình với ý kiến khác: xét nội dung được đồng tình.
  Có cả khen và chê: chọn ý chiếm ưu thế.
Phím khác: s = bỏ qua, b = quay lại mẫu trước, q = lưu và thoát."""


def label_sheet(rnd: int, inp=input, out=print) -> int:
    """Interactive labelling of round N in the terminal. Shows only the text (no current label, no model
    guess) and saves the sheet after every answer, so it can be stopped and resumed. Returns the number
    of items still unlabelled."""
    path = REVIEW_DIR / f"round_{rnd}.csv"
    sheet = pd.read_csv(path, dtype=str, encoding=CSV_ENCODING, keep_default_na=False)
    out(GUIDE)
    todo = [i for i in range(len(sheet)) if sheet.at[i, "my_label"].strip() == ""]
    pos, history = 0, []
    while pos < len(todo):
        i = todo[pos]
        done = int((sheet["my_label"].str.strip() != "").sum())
        out(f"\n{'=' * 70}\n[{done + 1}/{len(sheet)}] id {sheet.at[i, 'id']}")
        if sheet.at[i, "parent_text"]:
            out(f"  ↳ trả lời bình luận: {sheet.at[i, 'parent_text']}\n")
        out(f"  {sheet.at[i, 'text']}")
        key = inp("nhãn (1 neg / 2 neu / 3 pos, s, b, q)> ").strip().lower()
        if key == "q":
            break
        if key == "b" and history:
            pos = todo.index(history.pop())
            sheet.at[todo[pos], "my_label"] = ""
            sheet.to_csv(path, index=False, encoding=CSV_ENCODING)
            continue
        if key == "s":
            pos += 1
            continue
        if key not in LABEL_KEYS:
            out("  phím không hợp lệ")
            continue
        sheet.at[i, "my_label"] = LABEL_KEYS[key]
        if "labeled_by" in sheet:
            sheet.at[i, "labeled_by"] = "human"
        sheet.to_csv(path, index=False, encoding=CSV_ENCODING)
        history.append(i)
        pos += 1
    left = int((sheet["my_label"].str.strip() == "").sum())
    out(f"\nđã lưu {path}; còn {left} mẫu chưa gán")
    return left


def apply_review(rnd: int, write: bool) -> dict:
    """Compare the reviewer's `my_label` column with dev.csv. With write=True the decisions replace the
    `label` column for those ids only; every other byte of dev.csv stays as it was."""
    sheet = pd.read_csv(REVIEW_DIR / f"round_{rnd}.csv", dtype=str, encoding=CSV_ENCODING, keep_default_na=False)
    key = pd.read_csv(REVIEW_DIR / f"round_{rnd}_key.csv", dtype=str, encoding=CSV_ENCODING, keep_default_na=False)
    sheet["my_label"] = sheet["my_label"].str.strip().str.lower()
    if "labeled_by" not in sheet:
        sheet["labeled_by"] = "human"
    todo = sheet[sheet["my_label"] == ""]
    bad = set(sheet["my_label"]) - set(CLASSES) - {""}
    if bad:
        raise ValueError(f"round {rnd}: my_label must be one of {CLASSES}, found {bad}")
    if len(todo):
        raise ValueError(f"round {rnd}: {len(todo)} items still have an empty my_label")
    dev = load_dev(TASK)
    current = dev.set_index("id")["label"]
    m = sheet.merge(key[["id", "group", "model_pred"]], on="id")
    m["label_now"] = m["id"].map(current)
    changed = m[m["my_label"] != m["label_now"]]
    summary = {
        "round": rnd, "checked": int(len(m)), "changed": int(len(changed)),
        "checked_by_group": m["group"].value_counts().to_dict(),
        "changed_by_group": changed["group"].value_counts().to_dict(),
        "direction": Counter(f"{a}->{b}" for a, b in zip(changed["label_now"], changed["my_label"])),
        "model_agrees_with_reviewer": int((m["model_pred"] == m["my_label"]).sum()),
        "labeled_by": sheet["labeled_by"].replace("", "unknown").value_counts().to_dict(),
    }
    for r in changed.itertuples():
        print(f"{r.id} [{r.group}] {r.label_now} -> {r.my_label}: {r.text[:120]}")
    if write and len(changed):
        dev.loc[dev["id"].isin(changed["id"]), "label"] = dev["id"].map(dict(zip(changed["id"], changed["my_label"])))
        dev.to_csv(task_dir(TASK) / "dev.csv", index=False, encoding=CSV_ENCODING)
    summary["direction"] = dict(summary["direction"])
    summary["written"] = bool(write and len(changed))
    (REVIEW_DIR / f"round_{rnd}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- corpus

LOCAL_CONFIGS = ["visobert_pre_text", "visobert_pre_ctx", "tfidf_lr_text", "tfidf_lr_ctx"]


def predict_corpus(config: str) -> Path:
    """Corpus scores of a local model (TF-IDF is refit on all of dev, C chosen by CV on dev)."""
    corpus = load_corpus(TASK)
    use_parent = config.endswith("ctx")
    if config.startswith("visobert_pre"):
        proba = visobert_proba(corpus["text"], corpus["parent_text"] if use_parent else None)
    elif config.startswith("tfidf_lr"):
        dev, y = load_dev_xy()
        model, _ = tfidf_fit(dev[["text", "parent_text"]], y, use_parent)
        proba = model.predict_proba(corpus[["text", "parent_text"]])
    else:
        raise ValueError(f"unknown config {config}; use one of {LOCAL_CONFIGS}")
    out = RESULT_DIR / f"corpus_scores_{config}.csv"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scores_frame(corpus["id"], proba).to_csv(out, index=False, encoding=CSV_ENCODING)
    return out


def label_corpus(scores_file: Path) -> dict:
    """data/tasks/sentiment/corpus_labels.csv: id, label, score, p_neg, p_neu, p_pos.
    label is the model's argmax, except for dev items, which keep their hand-checked label.
    score = p_pos - p_neg (polarity in [-1, 1]) is always the model's, also for dev items."""
    corpus = load_corpus(TASK)
    s = pd.read_csv(scores_file, dtype={"id": str}, encoding=CSV_ENCODING)
    if set(s["id"]) != set(corpus["id"]) or len(s) != len(corpus):
        raise ValueError(f"{scores_file}: ids do not match corpus.csv")
    s = s.set_index("id").loc[corpus["id"]].reset_index()
    p = proba_of(s)
    label = pd.Series([CLASSES[k] for k in p.argmax(1)], index=s["id"])
    hand = dev_to_corpus_ids().set_index("corpus_id")["label"]
    n_overridden = int((label.loc[hand.index] != hand).sum())
    label.loc[hand.index] = hand
    out = pd.DataFrame({"id": s["id"], "label": label.to_numpy(), "score": np.round(p[:, 2] - p[:, 0], 4),
                        **{c: np.round(p[:, k], 4) for k, c in enumerate(PROBA_COLS)}})
    path = task_dir(TASK) / "corpus_labels.csv"
    out.to_csv(path, index=False, encoding=CSV_ENCODING)
    return {"file": str(path), "scores_file": str(scores_file), "n": int(len(out)),
            "dev_items_with_hand_label": int(len(hand)), "hand_label_differs_from_model": n_overridden,
            "label_share": out["label"].value_counts(normalize=True).round(3).to_dict()}


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sentiment model ladder on the dev set")
    sub = parser.add_subparsers(dest="action")
    p = sub.add_parser("ladder")
    p.add_argument("--skip-visobert", action="store_true")
    sub.add_parser("collect")
    p = sub.add_parser("review")
    p.add_argument("--scores", default="dev_scores_visobert_pre_text.csv",
                   help="dev scores file of the best current model (in results/sentiment/ or kaggle/)")
    p.add_argument("--n", type=int, default=N_PER_GROUP)
    p = sub.add_parser("label", help="label a review round interactively in the terminal")
    p.add_argument("round", type=int)
    p = sub.add_parser("apply-review")
    p.add_argument("round", type=int)
    p.add_argument("--write", action="store_true", help="write the decisions into dev.csv")
    p = sub.add_parser("predict-corpus")
    p.add_argument("config", choices=LOCAL_CONFIGS)
    p = sub.add_parser("label-corpus")
    p.add_argument("scores_file")
    args = parser.parse_args(argv)

    if args.action == "collect":
        cols = ["config", "source", "trained_on", "macro_f1", "macro_f1_ci95", "f1_neg", "f1_neu", "f1_pos",
                "macro_f1_top_level", "macro_f1_reply"]
        full = collect()
        print("dev labels", full["dev_labels_sha"].iloc[0])
        print(full[cols].sort_values("macro_f1", ascending=False).to_string(index=False))
    elif args.action == "review":
        f = Path(args.scores)
        if not f.exists():
            f = next((d / args.scores for d in (RESULT_DIR, KAGGLE_RESULT_DIR) if (d / args.scores).exists()), f)
        print(f"wrote {review(f, args.n)} (fill my_label with neg/neu/pos, then run apply-review)")
    elif args.action == "label":
        left = label_sheet(args.round)
        if not left:
            print(f"xong; tiếp theo: python -m src.classifiers.sentiment apply-review {args.round}")
    elif args.action == "apply-review":
        s = apply_review(args.round, args.write)
        print(json.dumps(s, indent=2, ensure_ascii=False))
        if not args.write and s["changed"]:
            print("dry run: add --write to change dev.csv")
    elif args.action == "predict-corpus":
        print(f"wrote {predict_corpus(args.config)}")
    elif args.action == "label-corpus":
        print(json.dumps(label_corpus(Path(args.scores_file)), indent=2, ensure_ascii=False))
    else:
        payload = run(getattr(args, "skip_visobert", False))
        cols = ["level", "config", "input", "macro_f1", "macro_f1_ci95", "f1_neg", "f1_neu", "f1_pos",
                "macro_f1_top_level", "macro_f1_reply"]
        print("dev labels", payload["dev_labels_sha"])
        print(pd.DataFrame(payload["results"])[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
