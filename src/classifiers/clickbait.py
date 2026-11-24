"""Clickbait model ladder on the dev set.

Levels run locally: baseline (majority class, keyword rules) and classical (TF-IDF + Logistic
Regression, Linear SVM, Complement Naive Bayes). PhoBERT and Qwen run on Kaggle
(notebooks/clickbait_*.ipynb) and report on the same dev items.

Data: data/tasks/clickbait/ (common.py). Training: train.csv (ViClickbait-2025, 3,414 headlines from 8
news sites). Evaluation: dev.csv (labelled VnExpress headlines). Models trained only on ViClickbait are
scored on all of dev; a model that also trains on dev is scored out of fold (5-fold CV over dev, the scored
fold never in training).

Usage (from the project root):
    python -m src.classifiers.clickbait ladder         # baselines, classical, input/training-data ablations
                                                       # (about 2 min on Apple M1, 20 min on an Intel i3 laptop)
    python -m src.classifiers.clickbait extras [--family tfidf_lr]  # extra baselines and ablations (needs pyvi)
    python -m src.classifiers.clickbait in-domain      # dev-trained models, nested CV (6 min M1, 14 min i3)
    python -m src.classifiers.clickbait kaggle-bundle  # input files for the Kaggle notebooks
    python -m src.classifiers.clickbait collect        # re-score Kaggle outputs, ensembles, one merged table
    python -m src.classifiers.clickbait final --config hand_tfidf_lr --in-domain dev   # corpus labels
    python -m src.classifiers.clickbait final --from-file results/clickbait/kaggle/corpus_scores_<config>.csv
    python -m src.classifiers.clickbait compare <config_a> <config_b>   # paired bootstrap on dev

Outputs (results/clickbait/): ladder (metrics.json, ladder.csv, ablation.csv), extras/ (extras.csv),
in_domain/ (in_domain.csv), ensemble/ and ladder_full.csv from collect, final.json; every configuration also
writes dev_scores_<config>.csv (id, pred, score; 4 decimals). Kaggle outputs use the same format and go to
results/clickbait/kaggle/.

extras: stratified random guess, handcrafted features + LR, decision threshold (0.5 / F1-optimal on
out-of-fold ViClickbait scores / prior shift to the VnExpress part of ViClickbait), word segmentation (pyvi),
class_weight, learning curve over the dev training folds, dev-only training.

final: labels for all of corpus.csv in data/tasks/clickbait/corpus_labels.csv (id, label, score = P(clickbait));
dev headlines (matched to corpus rows by title) keep their dev label. Run log in results/clickbait/final.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from .common import (CSV_ENCODING, RESULTS_DIR, ROOT, SEED, binary_metrics, dev_to_corpus_ids, evaluate_binary,
                     f1_threshold, hardware, load_corpus, load_dev, load_train, paired_bootstrap, round_scores,
                     run_info, task_dir, write_results)

RESULT_DIR = RESULTS_DIR / "clickbait"
EXTRAS_DIR = RESULT_DIR / "extras"
IN_DOMAIN_DIR = RESULT_DIR / "in_domain"
KAGGLE_RESULTS = RESULT_DIR / "kaggle"
ENSEMBLE_DIR = RESULT_DIR / "ensemble"
SCORE_DIRS = [RESULT_DIR, KAGGLE_RESULTS, EXTRAS_DIR, IN_DOMAIN_DIR, ENSEMBLE_DIR]
KAGGLE_DIR = ROOT / "data" / "kaggle" / "clickbait"
CORPUS_LABELS_PATH = task_dir("clickbait") / "corpus_labels.csv"

# Hand-written keyword rules for curiosity gaps and sensational wording, not tuned on dev.
KEYWORDS = [
    r"\?", r"!", r"\bsốc\b", r"bất ngờ", r"\blý do\b", r"\bvì sao\b", r"\btại sao\b", r"\bđiều gì\b",
    r"\bđiều này\b", r"\bngười này\b", r"\bbí mật\b", r"\bsự thật\b", r"\bchấn động\b", r"\brúng động\b",
    r"\bxôn xao\b", r"\bgây bão\b", r"\bai ngờ\b", r"\bhé lộ\b", r"\bngỡ ngàng\b", r"\bchoáng\b",
    r"\brùng mình\b", r"\bbật khóc\b", r"không thể tin", r"\bbí ẩn\b", r"\bcực hot\b", r"\bkhủng\b",
    r"\bnóng bỏng\b", r"\bác mộng\b", r"\btan hoang\b", r"\bđiều bạn cần biết\b",
]
KEYWORD_RE = [re.compile(k, re.IGNORECASE) for k in KEYWORDS]
QUOTES = "'\"‘’“”"


def keyword_score(texts: pd.Series) -> np.ndarray:
    return np.array([sum(bool(r.search(t)) for r in KEYWORD_RE) for t in texts], dtype=float)


def load_viclickbait() -> pd.DataFrame:
    df = load_train("clickbait")
    df["y"] = df["label"].astype(int)
    return df


def load_dev_y() -> tuple[pd.DataFrame, np.ndarray]:
    dev = load_dev("clickbait")
    return dev, dev["label"].astype(int).to_numpy()


def text_of(df: pd.DataFrame, use_sapo: bool) -> pd.Series:
    title = df["title"].fillna("").astype(str)
    return (title + " . " + df["sapo"].fillna("").astype(str)) if use_sapo else title


def features() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])


MODELS = {
    "tfidf_lr": (LogisticRegression(max_iter=2000, random_state=SEED), {"clf__C": [0.3, 1, 3, 10]}),
    "tfidf_svm": (LinearSVC(random_state=SEED), {"clf__C": [0.03, 0.1, 0.3, 1]}),
    "tfidf_nb": (ComplementNB(), {"clf__alpha": [0.1, 0.3, 1.0]}),
}


def make_pipe(family: str, balanced: bool = False) -> Pipeline:
    clf = clone(MODELS[family][0])
    if balanced:
        clf.set_params(class_weight="balanced")
    return Pipeline([("feat", features()), ("clf", clf)])


def grid_fit(pipe: Pipeline, grid: dict, x, y):
    """Grid search by 5-fold CV on the training data only (F1 of the clickbait class)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=1)
    search.fit(x, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def raw_score(model, x) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x)


def default_threshold(model) -> float:
    return 0.5 if hasattr(model, "predict_proba") else 0.0


def score_and_pred(model, x) -> tuple[np.ndarray, np.ndarray]:
    s = raw_score(model, x)
    return s, (s >= default_threshold(model)).astype(int)


def dev_folds(y: np.ndarray) -> list:
    return list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def oof_plus_dev(family: str, params: dict, xv, yv, xd, y, frac: float = 1.0):
    """ViClickbait + a share `frac` of the dev training folds for training, the held-out dev fold scored;
    repeated over the 5 dev folds. Returns out-of-fold scores, predictions and the mean dev items used."""
    score, pred, n_used = np.zeros(len(y)), np.zeros(len(y), dtype=int), []
    for tr, te in dev_folds(y):
        if frac < 1:
            tr, _ = train_test_split(tr, train_size=frac, stratify=y[tr], random_state=SEED)
        n_used.append(len(tr))
        pipe = make_pipe(family).set_params(**params).fit(pd.concat([xv, xd.iloc[tr]]), np.concatenate([yv, y[tr]]))
        score[te], pred[te] = score_and_pred(pipe, xd.iloc[te])
    return score, pred, int(np.mean(n_used))


def scores_frame(ids, pred, score) -> pd.DataFrame:
    return pd.DataFrame({"id": ids, "pred": np.asarray(pred, int), "score": round_scores(score)})


def ladder() -> dict:
    vc = load_viclickbait()
    dev, y = load_dev_y()
    yv = vc["y"].to_numpy()
    rows, per_config, fitted = [], {}, {}

    def add(level, model, train_name, use_sapo, score, pred, extra=None):
        r = {"level": level, "model": model, "train": train_name, "input": "title+sapo" if use_sapo else "title"}
        r.update(evaluate_binary(y, score, pred))
        r.update(extra or {})
        rows.append(r)
        name = f"{model}_{train_name}_{r['input']}".replace("+", "_plus_").replace("(oof)", "_oof").replace("-", "")
        per_config[f"dev_scores_{name}.csv"] = scores_frame(dev["id"], pred, score)

    # Baselines
    add("baseline", "majority", "-", False, np.zeros(len(dev)), np.zeros(len(dev), dtype=int))
    kw = keyword_score(dev["title"])
    add("baseline", "keywords", "-", False, kw, (kw >= 1).astype(int))

    # Classical, trained on all of ViClickbait, title only (main setting) and title + sapo
    for use_sapo in (False, True):
        for name, (_, grid) in MODELS.items():
            model, params, cv_f1 = grid_fit(make_pipe(name), grid, text_of(vc, use_sapo), yv)
            add("classical", name, "viclickbait", use_sapo, *score_and_pred(model, text_of(dev, use_sapo)),
                {"params": params, "viclickbait_cv_f1": round(cv_f1, 3)})
            fitted[(name, use_sapo)] = (model, params)

    # Training-data ablation with the best classical family (by dev PR-AUC, title only)
    main = pd.DataFrame(rows).query("level == 'classical' and input == 'title'")
    best = main.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    vne = vc[vc["source"] == "VnExpress"]
    for use_sapo in (False, True):
        _, params = fitted[(best, use_sapo)]
        m2, p2, cv2 = grid_fit(make_pipe(best), MODELS[best][1], text_of(vne, use_sapo), vne["y"].to_numpy())
        add("ablation", best, "viclickbait_vnexpress_only", use_sapo, *score_and_pred(m2, text_of(dev, use_sapo)),
            {"params": p2, "viclickbait_cv_f1": round(cv2, 3)})
        s, p, _ = oof_plus_dev(best, params, text_of(vc, use_sapo), yv, text_of(dev, use_sapo), y)
        add("ablation", best, "viclickbait+vnexpress_dev(oof)", use_sapo, s, p, {"params": params})

    table = pd.DataFrame(rows)
    ablation = table[(table.model == best) & (table.level.isin(["classical", "ablation"]))]

    # Share of all collected headlines predicted clickbait by the best ViClickbait model
    titles = load_corpus("clickbait")
    _, p_all = score_and_pred(fitted[(best, False)][0], text_of(titles, False))
    share = {"model": best, "train": "viclickbait", "input": "title", "share": round(float(p_all.mean()), 3),
             "n_articles": int(len(titles))}

    payload = {
        **run_info(), "task": "clickbait", "eval_set": "data/tasks/clickbait/dev.csv", "hardware": hardware(),
        "dev_positive_share": round(float(y.mean()), 3), "dev_n": int(len(y)),
        "viclickbait_n": int(len(vc)), "viclickbait_positive_share": round(float(yv.mean()), 3),
        "best_classical_by_dev_pr_auc": best, "corpus_predicted_share": share,
        "results": table.to_dict(orient="records"),
    }
    write_results("clickbait", payload, {"ladder.csv": table[table.level != "ablation"],
                                         "ablation.csv": ablation, **per_config})
    return payload


# ---------------------------------------------------------------------------------------------------------
# Kaggle levels (PhoBERT, Qwen): input bundle and re-scoring of the downloaded outputs

def kaggle_bundle() -> None:
    """Files to upload as a private Kaggle dataset: ViClickbait-2025, the dev headlines with their labels and
    all corpus headlines (to label with a Kaggle model). Headlines and lead paragraphs are public article
    text; no comments."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    vc = load_viclickbait()
    vc[["id", "title", "sapo", "source", "y"]].to_csv(KAGGLE_DIR / "viclickbait.csv", index=False)
    dev, y = load_dev_y()
    dev.assign(y=y)[["id", "title", "sapo", "y"]].to_csv(KAGGLE_DIR / "dev_headlines.csv", index=False)
    corpus = load_corpus("clickbait")
    corpus[["id", "title", "sapo"]].to_csv(KAGGLE_DIR / "corpus_headlines.csv", index=False)
    print(f"wrote {len(vc)} + {len(dev)} + {len(corpus)} rows to {KAGGLE_DIR}")


def kaggle_level(config: str) -> str:
    if "bert" in config:
        return "deep_learning"
    if config.startswith("qwen"):
        return "llm"
    return "other"


def read_dev_scores(path: Path, y_by_id: dict) -> pd.DataFrame:
    """A dev_scores file (columns id, pred, score in any order), checked against the dev ids, with y added."""
    k = pd.read_csv(path, dtype={"id": str}, encoding=CSV_ENCODING)
    if len(k) != len(y_by_id) or set(k["id"]) != set(y_by_id):
        raise ValueError(f"{path.name}: item ids do not match the dev set")
    k["y"] = k["id"].map(y_by_id)
    return k


def collect() -> pd.DataFrame:
    """Score every Kaggle output on the current dev labels with the same metrics, add ensembles, and join with
    the local ladder, extras and in-domain tables (each only if present) into results/clickbait/ladder_full.csv.

    Kaggle files (results/clickbait/kaggle/):
        dev_scores_<config>.csv          one row each; level deep_learning (*bert*), llm (qwen*)
        dev_scores_<config>_tuned.csv    threshold chosen on held-out ViClickbait in the notebook; scored as is
        dev_scores_<config>_seed<k>.csv  per-seed scores: give per-config mean and std of F1 and PR-AUC
    Ensembles (results/clickbait/ensemble/): mean of the two probabilities, pred at 0.5, for each PhoBERT dev
    score file with (a) TF-IDF + LR trained on ViClickbait (ladder) and (b) the final model's out-of-fold dev
    scores (in-domain, TF-IDF + handcrafted features + LR trained on dev)."""
    dev, y = load_dev_y()
    y_by_id = dict(zip(dev["id"], y))
    files = sorted(KAGGLE_RESULTS.glob("dev_scores_*.csv")) if KAGGLE_RESULTS.is_dir() else []
    seed_re = re.compile(r"^(?P<config>.+)_seed(?P<seed>\d+)$")
    rows, scores, seed_files = [], {}, {}
    for f in files:
        config = f.stem[len("dev_scores_"):]
        m = seed_re.match(config)
        if m:
            seed_files.setdefault(m["config"], []).append(f)
            continue
        k = read_dev_scores(f, y_by_id)
        scores[config] = k.set_index("id")
        rows.append({"level": kaggle_level(config), "model": config, "train": "see notebook", "input": "title",
                     **evaluate_binary(k["y"], k["score"], k["pred"]),
                     "note": "threshold tuned on held-out ViClickbait" if config.endswith("_tuned") else ""})

    # Mean and std over seeds, re-scored on the current dev labels
    for config, fs in seed_files.items():
        ms = [binary_metrics(k["y"], k["score"], k["pred"]) for k in (read_dev_scores(f, y_by_id) for f in fs)]
        st = {"seeds": len(ms)}
        for key in ("f1", "pr_auc"):
            v = [m[key] for m in ms]
            st[f"{key}_seed_mean"] = round(float(np.mean(v)), 3)
            st[f"{key}_seed_std"] = round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 3)
        for r in rows:
            if r["model"] == config:
                r.update(st)

    # Ensembles: probability mean of a local model and each PhoBERT configuration
    bases = {"tfidf_lr": RESULT_DIR / "dev_scores_tfidf_lr_viclickbait_title.csv",
             "hand_tfidf_lr_dev_oof": IN_DOMAIN_DIR / "dev_scores_hand_tfidf_lr_dev_oof.csv"}
    ens_files = {}
    for base, path in bases.items():
        if not path.exists():
            continue
        t = read_dev_scores(path, y_by_id).set_index("id").loc[dev["id"]]
        for config, k in scores.items():
            if "bert" not in config or config.endswith("_tuned"):
                continue
            mean = (t["score"].to_numpy() + k.loc[dev["id"], "score"].to_numpy()) / 2
            pred = (mean >= 0.5).astype(int)
            name = f"{base}+{config}"
            rows.append({"level": "ensemble", "model": name, "train": "see parts", "input": "title",
                         **evaluate_binary(y, mean, pred),
                         "note": f"{path.name} + dev_scores_{config}.csv; probability mean, pred at 0.5"})
            ens_files[f"dev_scores_{name}.csv"] = scores_frame(dev["id"], pred, mean)
    if ens_files:
        ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
        for fname, df in ens_files.items():
            df.to_csv(ENSEMBLE_DIR / fname, index=False, encoding=CSV_ENCODING)

    parts = []
    for p in (RESULT_DIR / "ladder.csv", EXTRAS_DIR / "extras.csv", IN_DOMAIN_DIR / "in_domain.csv"):
        if p.exists():
            parts.append(pd.read_csv(p, encoding=CSV_ENCODING))
        else:
            print(f"note: {p.relative_to(ROOT)} not found; run the matching subcommand to include it")
    if rows:
        parts.append(pd.DataFrame(rows))
    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["level", "model", "train", "input", "f1", "f1_ci95", "pr_auc", "pr_auc_ci95", "predicted_positive_share"])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(RESULT_DIR / "ladder_full.csv", index=False, encoding=CSV_ENCODING)
    return full


# ---------------------------------------------------------------------------------------------------------
# Extra baselines and ablations

def segment(texts: pd.Series) -> pd.Series:
    """Vietnamese word segmentation with pyvi (multi-syllable words joined by "_")."""
    from pyvi import ViTokenizer
    return texts.map(ViTokenizer.tokenize)


def handcrafted(texts) -> np.ndarray:
    """Length in characters and words, has ? and !, digit count, quote marks, colon, share of capitalised
    words, number of keyword rules matched."""
    texts = pd.Series(list(texts), dtype=str)
    rows = []
    for t in texts:
        words = t.split()
        rows.append([len(t), len(words), "?" in t, "!" in t, sum(c.isdigit() for c in t),
                     sum(t.count(q) for q in QUOTES), ":" in t,
                     (sum(w[:1].isupper() for w in words) / len(words)) if words else 0.0])
    return np.column_stack([np.array(rows, dtype=float), keyword_score(texts)])


def oof_train_scores(pipe: Pipeline, x, y) -> np.ndarray:
    """Out-of-fold scores on the training data (5-fold), for choosing a threshold inside training."""
    method = "predict_proba" if hasattr(pipe, "predict_proba") else "decision_function"
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    s = cross_val_predict(pipe, x, y, cv=cv, method=method, n_jobs=1)
    return s[:, 1] if s.ndim == 2 else s


def prior_shift(p: np.ndarray, src: float, tgt: float) -> np.ndarray:
    """Rescale probabilities from training prior src to target prior tgt (odds adjustment as in Saerens et
    al. 2002, with the target prior given rather than estimated by EM)."""
    a = (tgt / src) * p
    b = ((1 - tgt) / (1 - src)) * (1 - p)
    return a / (a + b)


def extras(family: str = "tfidf_lr") -> dict:
    vc = load_viclickbait()
    dev, y = load_dev_y()
    yv = vc["y"].to_numpy()
    xv, xd = text_of(vc, False), text_of(dev, False)
    rate = float(yv.mean())
    vne_rate = float(vc.loc[vc["source"] == "VnExpress", "y"].mean())
    grid = MODELS[family][1]
    rows, per_config = [], {}

    def add(level, group, config, variant, score, pred, seconds, extra=None):
        extra = dict(extra or {})
        r = {"level": level, "group": group, "model": config, "variant": variant,
             "train": extra.pop("train", "viclickbait"), "input": "title"}
        r.update(evaluate_binary(y, score, pred))
        r["seconds"] = round(float(seconds), 1)
        r.update(extra)
        rows.append(r)
        per_config[f"dev_scores_{config}.csv"] = scores_frame(dev["id"], pred, score)
        print(f"  {config:28s} f1={r['f1']:.3f} pr_auc={r['pr_auc']:.3f} share={r['predicted_positive_share']:.3f}"
              f" ({r['seconds']}s)", flush=True)

    # a. Baselines
    t0 = time.time()
    s = np.random.default_rng(SEED).random(len(dev))
    add("baseline", "baseline", "random_stratified", f"pred = U(0,1) < {rate:.3f} (ViClickbait rate)",
        s, (s < rate).astype(int), time.time() - t0, {"train": "-"})

    t0 = time.time()
    hand = Pipeline([("feat", FunctionTransformer(handcrafted)), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
    m, params, cv_f1 = grid_fit(hand, {"clf__C": [0.01, 0.1, 1, 10]}, xv, yv)
    add("baseline", "baseline", "handcrafted_lr", "9 handcrafted features + StandardScaler + LR",
        *score_and_pred(m, xd), time.time() - t0, {"params": params, "viclickbait_cv_f1": round(cv_f1, 3)})

    # b. Decision threshold (one fit, three rules)
    t0 = time.time()
    base, bparams, bcv = grid_fit(make_pipe(family), grid, xv, yv)
    fit_sec = time.time() - t0
    s0, p0 = score_and_pred(base, xd)
    thr0 = default_threshold(base)
    add("ablation", "threshold", f"{family}_thr_0.5", "threshold 0.5 (0 for SVM margin)", s0, p0,
        fit_sec, {"params": bparams, "viclickbait_cv_f1": round(bcv, 3), "threshold": thr0})
    t0 = time.time()
    thr = f1_threshold(yv, oof_train_scores(clone(base), xv, yv))
    add("ablation", "threshold", f"{family}_thr_cv", "F1-optimal threshold on out-of-fold ViClickbait scores",
        s0, (s0 >= thr).astype(int), fit_sec + time.time() - t0, {"params": bparams, "threshold": round(thr, 4)})
    if hasattr(base, "predict_proba"):
        sp = prior_shift(s0, rate, vne_rate)
        add("ablation", "threshold", f"{family}_thr_prior",
            f"prior shift {rate:.3f} -> {vne_rate:.3f} (VnExpress part of ViClickbait), threshold 0.5",
            sp, (sp >= 0.5).astype(int), fit_sec,
            {"params": bparams, "threshold": 0.5, "equivalent_raw_threshold": round(float(
                prior_shift(np.array([0.5]), vne_rate, rate)[0]), 4)})

    # c. Word segmentation
    t0 = time.time()
    xv_seg, xd_seg = segment(xv), segment(xd)
    mseg, pseg, cvseg = grid_fit(make_pipe(family), grid, xv_seg, yv)
    add("ablation", "segmentation", f"{family}_seg", "pyvi word segmentation", *score_and_pred(mseg, xd_seg),
        time.time() - t0, {"params": pseg, "viclickbait_cv_f1": round(cvseg, 3)})

    # d. Class balancing
    if "class_weight" in MODELS[family][0].get_params():
        t0 = time.time()
        mbal, pbal, cvbal = grid_fit(make_pipe(family, balanced=True), grid, xv, yv)
        add("ablation", "class_weight", f"{family}_balanced", 'class_weight="balanced"', *score_and_pred(mbal, xd),
            time.time() - t0, {"params": pbal, "viclickbait_cv_f1": round(cvbal, 3)})

    # e. Learning curve: ViClickbait + a share of the dev training folds, scored out of fold
    for frac in (0.25, 0.5, 0.75, 1.0):
        t0 = time.time()
        score, pred, n_used = oof_plus_dev(family, bparams, xv, yv, xd, y, frac)
        add("ablation", "learning_curve", f"{family}_lc_{int(frac * 100):03d}",
            f"ViClickbait + {int(frac * 100)}% of dev training folds (oof)", score, pred, time.time() - t0,
            {"params": bparams, "train": f"viclickbait+{int(frac * 100)}%dev(oof)", "dev_train_items_per_fold": n_used})

    # f. dev only, out of fold, hyperparameters by inner CV in each training fold
    t0 = time.time()
    score, pred, fold_params = np.zeros(len(dev)), np.zeros(len(dev), dtype=int), []
    for tr, te in dev_folds(y):
        m, fp, _ = grid_fit(make_pipe(family), grid, xd.iloc[tr], y[tr])
        score[te], pred[te] = score_and_pred(m, xd.iloc[te])
        fold_params.append(fp)
    add("ablation", "training_data", f"{family}_devonly_oof", "dev only (oof, inner 5-fold CV for C)", score, pred,
        time.time() - t0, {"params": fold_params, "train": "dev(oof)"})

    table = pd.DataFrame(rows)
    payload = {
        **run_info(), "task": "clickbait", "eval_set": "data/tasks/clickbait/dev.csv", "family": family,
        "hardware": hardware(), "dev_n": int(len(y)), "dev_positive_share": round(float(y.mean()), 3),
        "viclickbait_positive_share": round(rate, 3), "viclickbait_vnexpress_positive_share": round(vne_rate, 3),
        "notes": ["seconds: wall-clock for the configuration; threshold rows reuse the threshold-0.5 model and "
                  "include its fit time",
                  "the threshold-0.5 row is also the 0% point of the learning curve and the no-segmentation and "
                  "class_weight=None reference",
                  "threshold rows share one model: PR-AUC identical for 0.5 and cv; the prior shift is monotonic so "
                  "ranking (PR-AUC) is unchanged up to ties"],
        "results": table.to_dict(orient="records"),
    }
    write_results("clickbait/extras", payload, {"extras.csv": table, **per_config})
    return payload


# ---------------------------------------------------------------------------------------------------------
# In-domain models: trained on dev (optionally plus ViClickbait), TF-IDF and/or handcrafted features.
# Scored out of fold; C and the decision threshold are chosen by an inner 5-fold CV over the dev part of
# each training fold (never on the scored fold).

KINDS = ("tfidf_lr", "hand_lr", "hand_tfidf_lr")
C_GRID = [0.3, 1, 3, 10, 30]


def kind_pipe(kind: str, C: float) -> Pipeline:
    hand = Pipeline([("hand", FunctionTransformer(handcrafted)), ("scale", StandardScaler())])
    feat = {"tfidf_lr": features(), "hand_lr": hand,
            "hand_tfidf_lr": FeatureUnion([("tfidf", features()), ("hand", hand)])}[kind]
    return Pipeline([("feat", feat), ("clf", LogisticRegression(C=C, max_iter=3000, random_state=SEED))])


def tune_in_domain(kind: str, xd: pd.Series, yd: np.ndarray, xe: pd.Series | None, ye: np.ndarray | None):
    """Pick C by inner 5-fold PR-AUC over the dev items given (extra ViClickbait data, if any, is always in
    training), then the F1-optimal threshold on the inner out-of-fold scores of that C."""
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(xd, yd))
    best = None
    for C in C_GRID:
        s = np.zeros(len(yd))
        for a, b in splits:
            s[b] = fit_in_domain(kind, xd.iloc[a], yd[a], xe, ye, C).predict_proba(xd.iloc[b])[:, 1]
        ap = float(average_precision_score(yd, s))
        if best is None or ap > best[0]:
            best = (ap, C, s)
    return {"C": best[1], "threshold": f1_threshold(yd, best[2]), "inner_pr_auc": round(best[0], 4)}


def fit_in_domain(kind: str, xd, yd, xe, ye, C: float) -> Pipeline:
    x = xd if xe is None else pd.concat([xe, xd])
    y = yd if ye is None else np.concatenate([ye, yd])
    return kind_pipe(kind, C).fit(x, y)


def in_domain() -> dict:
    vc = load_viclickbait()
    dev, y = load_dev_y()
    xd, xv, yv = text_of(dev, False), text_of(vc, False), vc["y"].to_numpy()
    rows, per_config = [], {}
    for kind in KINDS:
        for train in ("dev", "viclickbait+dev"):
            t0 = time.time()
            xe, ye = (None, None) if train == "dev" else (xv, yv)
            score, pred, chosen = np.zeros(len(y)), np.zeros(len(y), dtype=int), []
            for tr, te in dev_folds(y):
                c = tune_in_domain(kind, xd.iloc[tr], y[tr], xe, ye)
                score[te] = fit_in_domain(kind, xd.iloc[tr], y[tr], xe, ye, c["C"]).predict_proba(xd.iloc[te])[:, 1]
                pred[te] = (score[te] >= c["threshold"]).astype(int)
                chosen.append(c)
            name = f"{kind}_{train.replace('viclickbait+', 'vc_plus_')}_oof"
            r = {"level": "classical", "group": "in_domain", "model": name, "train": f"{train}(oof)", "input": "title",
                 **evaluate_binary(y, score, pred), "seconds": round(time.time() - t0, 1),
                 "C_per_fold": [c["C"] for c in chosen], "threshold_per_fold": [round(c["threshold"], 3) for c in chosen]}
            rows.append(r)
            per_config[f"dev_scores_{name}.csv"] = scores_frame(dev["id"], pred, score)
            print(f"  {name:32s} f1={r['f1']:.3f} {r['f1_ci95']} pr_auc={r['pr_auc']:.3f} {r['pr_auc_ci95']} "
                  f"share={r['predicted_positive_share']:.3f} ({r['seconds']}s)", flush=True)
    table = pd.DataFrame(rows)
    payload = {**run_info(), "task": "clickbait", "hardware": hardware(), "C_grid": C_GRID,
               "protocol": "outer 5-fold over dev; inner 5-fold over the dev part of the training fold picks C "
                           "(PR-AUC) and the F1-optimal threshold", "results": table.to_dict(orient="records")}
    write_results("clickbait/in_domain", payload, {"in_domain.csv": table, **per_config})
    return payload


# ---------------------------------------------------------------------------------------------------------
# Final labels for the whole corpus

def final(config: str | None = None, from_file: str | None = None, in_domain_train: str = "dev") -> dict:
    """Retrain an in-domain model (`config` in KINDS) on all of dev (or ViClickbait + dev), or read corpus
    scores produced elsewhere (`from_file`: id, pred, score), and write corpus_labels.csv."""
    t0 = time.time()
    corpus = load_corpus("clickbait")
    dev, yd = load_dev_y()
    if from_file:
        k = pd.read_csv(from_file, dtype={"id": str})
        if not {"id", "score", "pred"} <= set(k.columns):
            raise ValueError(f"{from_file}: needs columns id, pred, score")
        if len(k) != len(corpus) or k["id"].duplicated().any() or set(k["id"]) != set(corpus["id"]):
            raise ValueError(f"{from_file}: ids do not match corpus.csv exactly")
        k = k.set_index("id").loc[corpus["id"]]
        score, pred = k["score"].to_numpy(float), k["pred"].astype(int).to_numpy()
        info = {"config": Path(from_file).name, "source_file": str(from_file), "threshold_rule": "from file"}
    else:
        if config not in KINDS:
            raise ValueError(f"--config must be one of {list(KINDS)} (or use --from-file)")
        xd = text_of(dev, False)
        xe, ye = (None, None)
        if in_domain_train == "viclickbait+dev":
            vc = load_viclickbait()
            xe, ye = text_of(vc, False), vc["y"].to_numpy()
        c = tune_in_domain(config, xd, yd, xe, ye)
        score = fit_in_domain(config, xd, yd, xe, ye, c["C"]).predict_proba(text_of(corpus, False))[:, 1]
        pred = (score >= c["threshold"]).astype(int)
        info = {"config": config, "train": in_domain_train, "C": c["C"], "threshold": round(c["threshold"], 4),
                "threshold_rule": "F1-optimal on inner 5-fold out-of-fold dev scores", "inner_pr_auc": c["inner_pr_auc"],
                "train_n": int(len(yd) + (0 if ye is None else len(ye))), "score": "probability"}

    out = pd.DataFrame({"id": corpus["id"].to_numpy(), "label": pred.astype(int).astype(str), "score": round_scores(score)})
    dev_labels = dev_to_corpus_ids("clickbait", keys=["title"]).set_index("corpus_id")["label"]
    is_dev = out["id"].isin(dev_labels.index).to_numpy()
    from_dev = out.loc[is_dev, "id"].map(dev_labels).to_numpy()
    agree = float((out.loc[is_dev, "label"].to_numpy() == from_dev).mean())
    out.loc[is_dev, "label"] = from_dev
    out.to_csv(CORPUS_LABELS_PATH, index=False, encoding=CSV_ENCODING)
    info.update({
        "n": int(len(out)), "n_dev_matched": int(is_dev.sum()),
        "model_positive_share_all": round(float(pred.mean()), 4),
        "label_positive_share_all": round(float((out["label"] == "1").mean()), 4),
        "model_positive_share_non_dev": round(float(pred[~is_dev].mean()), 4),
        "model_positive_share_dev_rows": round(float(pred[is_dev].mean()), 4),
        "dev_label_positive_share": round(float(yd.mean()), 4),
        "model_agrees_with_dev_label": round(agree, 4),
        "seconds": round(time.time() - t0, 1), "hardware": hardware(), "output": str(CORPUS_LABELS_PATH.relative_to(ROOT)),
    })
    payload = {**run_info(), "task": "clickbait", **info}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "final.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------------------------------------
# Paired comparison of two configurations on dev

def find_scores(config: str) -> Path:
    """dev_scores_<config>.csv in results/clickbait/ or its kaggle/, extras/, in_domain/, ensemble/ folders
    (or a path to such a file)."""
    if Path(config).is_file():
        return Path(config)
    name = config if config.startswith("dev_scores_") else f"dev_scores_{config}"
    name = name if name.endswith(".csv") else f"{name}.csv"
    hits = [d / name for d in SCORE_DIRS if (d / name).exists()]
    if not hits:
        raise FileNotFoundError(f"{name} not found in {[str(d.relative_to(ROOT)) for d in SCORE_DIRS]}")
    if len(hits) > 1:
        raise ValueError(f"{name} found in several folders; pass a path: {[str(h.relative_to(ROOT)) for h in hits]}")
    return hits[0]


def compare(a: str, b: str) -> dict:
    """Paired bootstrap (same resampled dev items for both) of F1 and PR-AUC, a minus b."""
    dev, y = load_dev_y()
    y_by_id = dict(zip(dev["id"], y))
    pa, pb = find_scores(a), find_scores(b)
    ka, kb = (read_dev_scores(p, y_by_id).set_index("id").loc[dev["id"]] for p in (pa, pb))
    f1 = lambda t, p: f1_score(t, p, zero_division=0)
    return {"a": str(pa.relative_to(ROOT)) if pa.is_absolute() else str(pa),
            "b": str(pb.relative_to(ROOT)) if pb.is_absolute() else str(pb), "n": int(len(y)),
            "f1_a": round(f1(y, ka["pred"]), 3), "f1_b": round(f1(y, kb["pred"]), 3),
            "pr_auc_a": round(float(average_precision_score(y, ka["score"])), 3),
            "pr_auc_b": round(float(average_precision_score(y, kb["score"])), 3),
            "delta_f1": paired_bootstrap(y, ka["pred"].to_numpy(), kb["pred"].to_numpy(), f1),
            "delta_pr_auc": paired_bootstrap(y, ka["score"].to_numpy(), kb["score"].to_numpy(), average_precision_score)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Clickbait models on the VnExpress dev set")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ladder", help="baselines, classical models and the input / training-data ablations")
    ex = sub.add_parser("extras", help="extra baselines and ablations (threshold, pyvi, class_weight, learning curve)")
    ex.add_argument("--family", default="tfidf_lr", choices=list(MODELS))
    sub.add_parser("in-domain", help="models trained on dev, scored out of fold with nested CV")
    sub.add_parser("kaggle-bundle", help="write the input files for the Kaggle notebooks to data/kaggle/clickbait/")
    sub.add_parser("collect", help="re-score Kaggle outputs, ensembles, merged table ladder_full.csv")
    fi = sub.add_parser("final", help="label corpus.csv into data/tasks/clickbait/corpus_labels.csv")
    src = fi.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", choices=list(KINDS), help="in-domain model to retrain")
    src.add_argument("--from-file", help="corpus scores file with columns id, pred, score (e.g. from Kaggle)")
    fi.add_argument("--in-domain", default="dev", choices=["dev", "viclickbait+dev"],
                    help="with --config: training data; C and threshold by 5-fold CV over dev (default dev)")
    co = sub.add_parser("compare", help="paired bootstrap of two configurations on dev (a minus b)")
    co.add_argument("a", help="config name (dev_scores_<config>.csv) or path")
    co.add_argument("b", help="config name (dev_scores_<config>.csv) or path")
    args = parser.parse_args(argv)
    show = ["model", "f1", "f1_ci95", "precision", "recall", "pr_auc", "pr_auc_ci95", "predicted_positive_share"]

    if args.cmd is None:
        parser.print_help()
        return 2
    if args.cmd == "ladder":
        t0 = time.time()
        payload = ladder()
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(pd.DataFrame(payload["results"])[["level", "train", "input", *show]].to_string(index=False))
        print("corpus predicted share:", payload["corpus_predicted_share"])
        print(f"total {time.time() - t0:.0f}s")
    elif args.cmd == "extras":
        t0 = time.time()
        payload = extras(args.family)
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(pd.DataFrame(payload["results"])[["group", *show, "seconds"]].to_string(index=False))
        print(f"hardware: {payload['hardware']}; total {time.time() - t0:.0f}s")
    elif args.cmd == "in-domain":
        t0 = time.time()
        payload = in_domain()
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(pd.DataFrame(payload["results"])[[*show, "seconds"]].to_string(index=False))
        print(f"total {time.time() - t0:.0f}s")
    elif args.cmd == "kaggle-bundle":
        kaggle_bundle()
    elif args.cmd == "collect":
        full = collect()
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(full[["level", "train", "input", "model", "f1", "f1_ci95", "pr_auc", "pr_auc_ci95",
                        "predicted_positive_share"]].to_string(index=False))
    elif args.cmd == "final":
        payload = final(args.config, args.from_file, args.in_domain)
        print(json.dumps({k: v for k, v in payload.items() if k != "hardware"}, indent=2, ensure_ascii=False))
    elif args.cmd == "compare":
        print(json.dumps(compare(args.a, args.b), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
