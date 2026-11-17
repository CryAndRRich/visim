"""Clickbait model ladder on the dev set.

Levels run locally: baseline (majority class, keyword rules) and classical (TF-IDF + Logistic
Regression, Linear SVM, Complement Naive Bayes). PhoBERT and Qwen run on Kaggle
(notebooks/clickbait_*.ipynb) and report on the same dev items.

Data: data/tasks/clickbait/ (common.py). Training: train.csv (ViClickbait-2025, 3,414 headlines from 8
news sites). Evaluation: dev.csv (labelled VnExpress headlines). Models trained only on ViClickbait are
scored on all of dev; a model that also trains on dev is scored out of fold.

Ablations (classical level, best model family):
    input     title only vs title + lead paragraph (sapo)
    training  ViClickbait (all sources) vs its VnExpress part only vs ViClickbait + VnExpress dev
              (scored out of fold: 5-fold CV over dev, the scored fold never in training)

Usage (from the project root):
    python -m src.classifiers.clickbait ladder         # local levels (about 10 minutes on CPU)
    python -m src.classifiers.clickbait kaggle-bundle  # input files for the Kaggle notebooks
    python -m src.classifiers.clickbait collect        # merge Kaggle outputs into one table
Output: results/clickbait/ (metrics.json, ladder.csv, ablation.csv, ladder_full.csv, and one
dev_scores_<config>.csv per configuration with columns id, pred, score). Kaggle outputs use the same
format and go to results/clickbait/kaggle/.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .common import RESULTS_DIR, ROOT, SEED, binary_metrics, bootstrap_ci, load_corpus, load_dev, load_train, run_info, write_results


# Hand-written keyword rules for curiosity gaps and sensational wording, not tuned on dev.
KEYWORDS = [
    r"\?", r"!", r"\bsốc\b", r"bất ngờ", r"\blý do\b", r"\bvì sao\b", r"\btại sao\b", r"\bđiều gì\b",
    r"\bđiều này\b", r"\bngười này\b", r"\bbí mật\b", r"\bsự thật\b", r"\bchấn động\b", r"\brúng động\b",
    r"\bxôn xao\b", r"\bgây bão\b", r"\bai ngờ\b", r"\bhé lộ\b", r"\bngỡ ngàng\b", r"\bchoáng\b",
    r"\brùng mình\b", r"\bbật khóc\b", r"không thể tin", r"\bbí ẩn\b", r"\bcực hot\b", r"\bkhủng\b",
    r"\bnóng bỏng\b", r"\bác mộng\b", r"\btan hoang\b", r"\bđiều bạn cần biết\b",
]
KEYWORD_RE = [re.compile(k, re.IGNORECASE) for k in KEYWORDS]


def keyword_score(texts: pd.Series) -> np.ndarray:
    return np.array([sum(bool(r.search(t)) for r in KEYWORD_RE) for t in texts], dtype=float)


def load_viclickbait() -> pd.DataFrame:
    df = load_train("clickbait")
    df["y"] = df["label"].astype(int)
    return df


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


def fit(name: str, x: pd.Series, y: np.ndarray):
    """Grid search by 5-fold CV on the training data only (F1 of the clickbait class)."""
    clf, grid = MODELS[name]
    pipe = Pipeline([("feat", features()), ("clf", clone(clf))])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=1)
    search.fit(x, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def score_and_pred(model, x: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(x)[:, 1]
        return s, (s >= 0.5).astype(int)
    s = model.decision_function(x)
    return s, (s >= 0).astype(int)


def evaluate(y, score, pred) -> dict:
    m = binary_metrics(y, score, pred)
    m.update(bootstrap_ci(y, score, pred))
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}


def out_of_fold_with_dev(name: str, params: dict, train: pd.DataFrame, dev: pd.DataFrame, use_sapo: bool):
    """ViClickbait + 4/5 of dev for training, the remaining fifth scored; repeated over 5 folds."""
    clf, _ = MODELS[name]
    score, pred = np.zeros(len(dev)), np.zeros(len(dev), dtype=int)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in folds.split(dev, dev["y"]):
        x = pd.concat([text_of(train, use_sapo), text_of(dev.iloc[tr], use_sapo)])
        y = np.concatenate([train["y"].to_numpy(), dev["y"].to_numpy()[tr]])
        pipe = Pipeline([("feat", features()), ("clf", clone(clf))])
        pipe.set_params(**params).fit(x, y)
        score[te], pred[te] = score_and_pred(pipe, text_of(dev.iloc[te], use_sapo))
    return score, pred


def run() -> dict:
    vc = load_viclickbait()
    dev = load_dev("clickbait")
    dev["y"] = dev["label"].astype(int)
    y = dev["y"].to_numpy()
    titles = load_corpus("clickbait")

    rows, per_config, fitted = [], {}, {}

    def add(level, model, train_name, use_sapo, score, pred, extra=None):
        r = {"level": level, "model": model, "train": train_name, "input": "title+sapo" if use_sapo else "title"}
        r.update(evaluate(y, score, pred))
        r.update(extra or {})
        rows.append(r)
        name = f"{model}_{train_name}_{r['input']}".replace("+", "_plus_").replace("(oof)", "_oof").replace("-", "")
        per_config[f"dev_scores_{name}.csv"] = pd.DataFrame({"id": dev["id"], "pred": pred, "score": np.round(score, 4)})

    # Baselines
    add("baseline", "majority", "-", False, np.zeros(len(dev)), np.zeros(len(dev), dtype=int))
    kw = keyword_score(dev["title"])
    add("baseline", "keywords", "-", False, kw, (kw >= 1).astype(int))

    # Classical, trained on all of ViClickbait, title only (main setting) and title + sapo
    for use_sapo in (False, True):
        for name in MODELS:
            model, params, cv_f1 = fit(name, text_of(vc, use_sapo), vc["y"].to_numpy())
            s, p = score_and_pred(model, text_of(dev, use_sapo))
            add("classical", name, "viclickbait", use_sapo, s, p,
                {"params": params, "viclickbait_cv_f1": round(cv_f1, 3)})
            fitted[(name, use_sapo)] = (model, params)
    ladder = pd.DataFrame(rows)

    # Training-data ablation with the best classical family (by dev PR-AUC, title only)
    main = ladder[(ladder.level == "classical") & (ladder.input == "title")]
    best = main.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    for use_sapo in (False, True):
        model, params = fitted[(best, use_sapo)]
        vne = vc[vc["source"] == "VnExpress"]
        m2, p2, cv2 = fit(best, text_of(vne, use_sapo), vne["y"].to_numpy())
        s, p = score_and_pred(m2, text_of(dev, use_sapo))
        add("ablation", best, "viclickbait_vnexpress_only", use_sapo, s, p, {"params": p2, "viclickbait_cv_f1": round(cv2, 3)})
        s, p = out_of_fold_with_dev(best, params, vc, dev, use_sapo)
        add("ablation", best, "viclickbait+vnexpress_dev(oof)", use_sapo, s, p, {"params": params})

    table = pd.DataFrame(rows)
    ablation = table[(table.model == best) & (table.level.isin(["classical", "ablation"]))]

    # Share of all 1,244 collected headlines predicted clickbait (sanity check against the public model's 84%)
    model, _ = fitted[(best, False)]
    _, p_all = score_and_pred(model, text_of(titles, False))
    share = {"model": best, "train": "viclickbait", "input": "title", "share": round(float(p_all.mean()), 3),
             "n_articles": int(len(titles))}

    payload = {
        **run_info(),
        "task": "clickbait", "eval_set": "data/tasks/clickbait/dev.csv",
        "dev_positive_share": round(float(y.mean()), 3), "dev_n": int(len(y)),
        "viclickbait_n": int(len(vc)), "viclickbait_positive_share": round(float(vc["y"].mean()), 3),
        "best_classical_by_dev_pr_auc": best, "corpus_predicted_share": share,
        "results": table.to_dict(orient="records"),
    }
    write_results("clickbait", payload, {"ladder.csv": table[table.level != "ablation"],
                                             "ablation.csv": ablation, **per_config})
    return payload


KAGGLE_DIR = ROOT / "data" / "kaggle" / "clickbait"
RESULT_DIR = RESULTS_DIR / "clickbait"


def kaggle_bundle() -> None:
    """Files to upload as a private Kaggle dataset: ViClickbait-2025 and the dev headlines with their
    labels. Headlines and lead paragraphs are public article text; no comments."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    vc = load_viclickbait()
    vc[["id", "title", "sapo", "source", "y"]].to_csv(KAGGLE_DIR / "viclickbait.csv", index=False)
    dev = load_dev("clickbait")
    dev["y"] = dev["label"].astype(int)
    dev[["id", "title", "sapo", "y"]].to_csv(KAGGLE_DIR / "dev_headlines.csv", index=False)
    print(f"wrote {len(vc)} + {len(dev)} rows to {KAGGLE_DIR}")


def collect() -> pd.DataFrame:
    """Score every Kaggle output on dev with the same metrics, and join with the local ladder."""
    dev = load_dev("clickbait")
    y_by_id = dict(zip(dev["id"], dev["label"].astype(int)))
    rows = []
    for f in sorted((RESULT_DIR / "kaggle").glob("dev_scores_*.csv")):
        k = pd.read_csv(f, dtype={"id": str})
        if set(k["id"]) != set(y_by_id):
            raise ValueError(f"{f.name}: item ids do not match the dev set")
        y = k["id"].map(y_by_id).to_numpy()
        config = f.stem[len("dev_scores_"):]
        level = "llm" if config.startswith("qwen") else "deep_learning"
        rows.append({"level": level, "model": config, "train": "see notebook", "input": "title",
                     **evaluate(y, k["score"].to_numpy(), k["pred"].to_numpy())})
    local = pd.read_csv(RESULT_DIR / "ladder.csv", encoding="utf-8-sig")
    full = pd.concat([local, pd.DataFrame(rows)], ignore_index=True)
    full.to_csv(RESULT_DIR / "ladder_full.csv", index=False, encoding="utf-8-sig")
    return full


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Clickbait model ladder on the dev set")
    parser.add_argument("action", nargs="?", default="ladder", choices=["ladder", "kaggle-bundle", "collect"])
    args = parser.parse_args(argv)
    if args.action == "kaggle-bundle":
        kaggle_bundle()
        return 0
    if args.action == "collect":
        full = collect()
        print(full[["level", "model", "train", "input", "f1", "f1_ci95", "pr_auc", "pr_auc_ci95",
                    "predicted_positive_share"]].to_string(index=False))
        return 0
    payload = run()
    cols = ["level", "model", "train", "input", "f1", "f1_ci95", "precision", "recall", "pr_auc", "pr_auc_ci95",
            "predicted_positive_share"]
    print(pd.DataFrame(payload["results"])[cols].to_string(index=False))
    print("corpus predicted share:", payload["corpus_predicted_share"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
