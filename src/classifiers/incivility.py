"""Evaluate incivility models on the VnExpress dev set and label the comment corpus.

Run from the repository root with `python -m src.classifiers.incivility --help`.
Model predictions are written to results/incivility. `collect` requires the score files and the
JSON sidecars (*_dev.json, *_corpus.json) written by the GPU notebooks in results/incivility/kaggle/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

from .common import (
    CSV_ENCODING, RESULTS_DIR, ROOT, SEED, binary_metrics, corpus_path, dev_to_corpus_ids, evaluate_binary,
    f1_threshold, load_corpus, load_dev, load_train, paired_bootstrap, round_scores, run_info, task_dir,
    write_results,
)

TASK = "incivility"
DATA = task_dir(TASK)
RESULT = RESULTS_DIR / TASK
KAGGLE = RESULT / "kaggle"
BUNDLE = ROOT / "data" / "kaggle" / TASK
FOLDS = 5
C_VALUES = (0.25, 1.0, 4.0)
SOURCES = ("victsd", "dev", "victsd_plus_dev")
INPUTS = ("comment", "parent")
THRESHOLDS = ("t05", "tf1")
INPUT_FILES = ("train", "dev", "corpus")
PHOBERT = "phobert_victsd_comment"
PHOBERT_SEEDS = (42, 43, 44)
QWEN_MODES = ("zero", "few", "cot")
GPU_CONFIGS = {
    "phobert": (PHOBERT,),
    "qwen": tuple(f"qwen_{mode}_parent" for mode in QWEN_MODES),
}

KEYWORDS = re.compile(
    r"\b(?:đọc lại bài đi|có hiểu gì không|có đọc bài không|nói như đúng rồi|"
    r"ngậm miệng|câm miệng|bớt ảo tưởng|không có não|não cá vàng|"
    r"ngu như bò|ngu ngốc|đồ ngu|đồ điên|thằng ngu|con ngu|"
    r"súc vật|cầm thú|chó má|óc chó|óc lợn|mất dạy|vô học|"
    r"đồ dối trá|toàn bịa đặt|bịa chuyện|nói láo|đồ gian dối|"
    r"đéo|đếch|địt|đụ má|đĩ mẹ|đm|đcm|dm|dcm|vcl|vl|clgt)\b"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def public_training(dev: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Exclude external rows whose comment also appears as a dev evaluation target."""
    train = load_train(TASK)
    duplicate = train["text"].map(normalized).isin(set(dev["text"].map(normalized)))
    return train.loc[~duplicate].reset_index(drop=True), int(duplicate.sum())


def model_input(frame: pd.DataFrame, with_parent: bool) -> pd.Series:
    text = frame["text"].map(normalized)
    if with_parent:
        parent = frame["parent_text"].map(normalized)
        # Give context its own features so it cannot be mistaken for the target comment.
        context = parent.map(lambda value: " ".join("parent_" + word for word in value.split()))
        text = (text + " " + context).str.rstrip()
    return text


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["label"].to_numpy(dtype=int)


def feature_union() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])


def logistic(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=SEED)


def fold_splits(y: np.ndarray):
    return StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(y, y)


def training_parts(train: pd.DataFrame, local: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "victsd":
        return train
    if source == "dev":
        return local
    return pd.concat([train, local], ignore_index=True)


def remove_held_out_parents(frame: pd.DataFrame, held_out: pd.DataFrame) -> pd.DataFrame:
    """A dev target must never appear in the training inputs, even as context."""
    blocked = set(held_out["text"].map(normalized))
    mask = frame["parent_text"].map(normalized).isin(blocked)
    return frame.assign(parent_text=frame["parent_text"].mask(mask, ""))


def fit_tfidf(train: pd.DataFrame, local: pd.DataFrame, source: str, with_parent: bool):
    """Tune C by PR-AUC, then tune the F1 threshold on the same training-only OOF scores."""
    tuning = train if source == "victsd" else local
    y = labels(tuning)
    out_of_fold = np.empty((len(C_VALUES), len(y)))
    for fit, val in fold_splits(y):
        subset = tuning.iloc[fit]
        fit_frame = subset if source == "victsd" else training_parts(train, subset, source)
        val_frame = tuning.iloc[val]
        if with_parent and source != "victsd":
            fit_frame = remove_held_out_parents(fit_frame, val_frame)
        features = feature_union()
        x_fit = features.fit_transform(model_input(fit_frame, with_parent))
        x_val = features.transform(model_input(val_frame, with_parent))
        for index, c in enumerate(C_VALUES):
            model = logistic(c).fit(x_fit, labels(fit_frame))
            out_of_fold[index, val] = model.predict_proba(x_val)[:, 1]

    aps = [average_precision_score(y, score) for score in out_of_fold]
    best = int(np.argmax(aps))
    c = C_VALUES[best]
    threshold = f1_threshold(y, out_of_fold[best])
    fitted = training_parts(train, local, source)
    features = feature_union()
    model = logistic(c).fit(features.fit_transform(model_input(fitted, with_parent)), labels(fitted))
    return features, model, {"C": c, "threshold_f1": threshold, "inner_pr_auc": float(aps[best])}


def tfidf_scores(fitted, frame: pd.DataFrame, with_parent: bool) -> np.ndarray:
    features, model, _ = fitted
    return model.predict_proba(features.transform(model_input(frame, with_parent)))[:, 1]


def evaluate_tfidf(train: pd.DataFrame, dev: pd.DataFrame, source: str, with_parent: bool):
    if source == "victsd":
        fitted = fit_tfidf(train, dev.iloc[:0], source, with_parent)
        score = tfidf_scores(fitted, dev, with_parent)
        return score, (score >= fitted[2]["threshold_f1"]).astype(int), [fitted[2]]

    score = np.empty(len(dev))
    tuned_pred = np.empty(len(dev), dtype=int)
    parameters = []
    for fit, val in fold_splits(labels(dev)):
        fit_frame, val_frame = dev.iloc[fit], dev.iloc[val]
        if with_parent:
            fit_frame = remove_held_out_parents(fit_frame, val_frame)
        fitted = fit_tfidf(train, fit_frame, source, with_parent)
        score[val] = tfidf_scores(fitted, val_frame, with_parent)
        tuned_pred[val] = score[val] >= fitted[2]["threshold_f1"]
        parameters.append(fitted[2])
    return score, tuned_pred, parameters


def baseline_scores(name: str, train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    if name == "majority":
        return np.full(len(frame), labels(train).mean())
    hits = frame["text"].map(lambda text: len(KEYWORDS.findall(normalized(text)))).to_numpy()
    return 1.0 - np.exp2(-hits.astype(float))


def prediction_table(frame: pd.DataFrame, score: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"id": frame["id"].to_numpy(), "pred": pred, "score": round_scores(score)})


def group_masks(dev: pd.DataFrame) -> dict[str, np.ndarray]:
    """dev ids 001-500 are a random corpus sample; E... and R... ids are strata oversampled for positives."""
    replies = dev["parent_text"].str.strip().ne("").to_numpy()
    ids = dev["id"]
    return {
        "all": np.ones(len(dev), dtype=bool), "root": ~replies, "reply": replies,
        "random": ids.str.isdigit().to_numpy(), "stratum_E": ids.str.startswith("E").to_numpy(),
        "stratum_R": ids.str.startswith("R").to_numpy(),
    }


def metric_rows(dev: pd.DataFrame, predictions: pd.DataFrame, details: dict) -> list[dict]:
    y = labels(dev)
    score = predictions["score"].to_numpy()
    pred = predictions["pred"].to_numpy()
    rows = []
    for group, mask in group_masks(dev).items():
        if not mask.any():
            continue
        group_y, group_s, group_p = y[mask], score[mask], pred[mask]
        two_classes = len(np.unique(group_y)) == 2
        metrics = evaluate_binary(group_y, group_s, group_p) if two_classes else binary_metrics(group_y, group_s, group_p)
        rows.append({**details, "group": group, "n": int(mask.sum()), "positives": int(group_y.sum()), **metrics})
    return rows


def tfidf_name(source: str, input_name: str, threshold: str) -> str:
    return f"tfidf_lr_{source}_{input_name}_{threshold}"


def timed_tfidf(train: pd.DataFrame, dev: pd.DataFrame, source: str, input_name: str):
    start = time.perf_counter()
    result = evaluate_tfidf(train, dev, source, input_name == "parent")
    return result, round(time.perf_counter() - start, 3)


def data_hashes() -> dict[str, str]:
    return {f"{name}_sha256": digest(corpus_path(TASK) if name == "corpus" else DATA / f"{name}.csv")
            for name in INPUT_FILES}


def local_ladder(initial: bool, jobs: int = 1) -> pd.DataFrame:
    hashes = data_hashes()
    dev = load_dev(TASK)
    train, duplicates = public_training(dev)
    rows, tables = [], {}

    def report(config: str, score, pred, **details):
        predictions = prediction_table(dev, score, pred)
        tables[f"dev_scores_{config}.csv"] = predictions
        rows.extend(metric_rows(dev, predictions, {"config": config, **details}))

    for name in ("majority", "keywords"):
        start = time.perf_counter()
        score = baseline_scores(name, train, dev)
        report(name, score, (score >= 0.5).astype(int), level="baseline", input="comment",
               train="victsd prior" if name == "majority" else "rules",
               threshold="0.5", seconds=round(time.perf_counter() - start, 3))

    settings = [("victsd", "comment")] if initial else [
        (source, input_name) for source in SOURCES for input_name in INPUTS
    ]
    evaluations = Parallel(n_jobs=min(jobs, len(settings)))(
        delayed(timed_tfidf)(train, dev, source, input_name) for source, input_name in settings
    )
    for (source, input_name), ((score, tuned_pred, parameters), elapsed) in zip(settings, evaluations):
        decisions = [("t05", (score >= 0.5).astype(int))]
        if not initial:
            decisions.append(("tf1", tuned_pred))
        for threshold_name, pred in decisions:
            report(tfidf_name(source, input_name, threshold_name), score, pred, level="classical",
                   input=input_name, train=source, threshold="0.5" if threshold_name == "t05" else "training F1",
                   seconds=elapsed, folds=parameters)

    table = pd.DataFrame(rows)
    directory = f"{TASK}/initial" if initial else TASK
    if data_hashes() != hashes:
        raise ValueError("task data changed during evaluation; rerun on the current CSV files")
    payload = {
        **run_info(), **hashes, "task": TASK,
        "dev_n": len(dev), "dev_positives": int(labels(dev).sum()), "train_n": len(train),
        "train_positives": int(labels(train).sum()), "train_rows_excluded": duplicates,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "architecture": platform.machine(), "cpu_jobs": min(jobs, len(settings)), "results": rows,
    }
    write_results(directory, payload, {"ladder.csv": table, **tables})
    return table


def read_predictions(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(path, dtype={"id": str})
    if list(predictions.columns) != ["id", "pred", "score"]:
        raise ValueError(f"{path.name}: expected id, pred, score columns")
    if predictions["id"].duplicated().any() or set(predictions["id"]) != set(frame["id"]):
        raise ValueError(f"{path.name}: predictions must cover every input ID exactly once")
    if not predictions["pred"].isin([0, 1]).all() or not predictions["score"].between(0, 1).all():
        raise ValueError(f"{path.name}: predictions must be 0 or 1 and scores must be in [0, 1]")
    return predictions.set_index("id").loc[frame["id"]].reset_index()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def kaggle_bundle() -> Path:
    """Build the input files for both GPU notebooks from this task's data."""
    hashes = data_hashes()
    train, duplicates = public_training(load_dev(TASK))
    BUNDLE.mkdir(parents=True, exist_ok=True)
    train_path = BUNDLE / "incivility_train.csv"
    train.to_csv(train_path, index=False, encoding=CSV_ENCODING)
    for name in ("dev", "corpus"):
        shutil.copyfile(corpus_path(TASK) if name == "corpus" else DATA / f"{name}.csv", BUNDLE / f"incivility_{name}.csv")
    if data_hashes() != hashes:
        raise ValueError("task data changed while building the GPU bundle")
    write_json(BUNDLE / "incivility_manifest.json", {
        **hashes, "bundle_train_sha256": digest(train_path), "train_rows_excluded": duplicates,
    })
    return BUNDLE


def gpu_run(family: str, phase: str) -> dict:
    path = KAGGLE / f"{family}_{phase}.json"
    run = read_json(path)
    if run["phase"] != phase or any(run["data"].get(key) != value for key, value in data_hashes().items()):
        raise ValueError(f"{path.name} was produced for different data or a different phase")
    return run


def read_gpu_predictions(run: dict, name: str, frame: pd.DataFrame) -> pd.DataFrame:
    path = KAGGLE / name
    if run["output_sha256"].get(name) != digest(path):
        raise ValueError(f"{name} does not match the corresponding notebook run")
    return read_predictions(path, frame)


def collect() -> pd.DataFrame:
    """Join complete notebook outputs to the CPU results, scoring every dev subgroup."""
    dev = load_dev(TASK)
    local = read_json(RESULT / "metrics.json")
    if any(local.get(key) != value for key, value in data_hashes().items()):
        raise ValueError("CPU ladder results do not match the current data")
    rows = local["results"]
    for family, configs in GPU_CONFIGS.items():
        run = gpu_run(family, "dev")
        if set(run["configs"]) != set(configs):
            raise ValueError(f"{family}_dev.json is missing a model configuration")
        for config in configs:
            predictions = read_gpu_predictions(run, f"dev_scores_{config}.csv", dev)
            rows.extend(metric_rows(dev, predictions, {
                "config": config, "level": "deep" if family == "phobert" else "llm",
                "input": "comment" if family == "phobert" else "parent",
                "train": "victsd" if family == "phobert" else "prompt", "threshold": "0.5",
                "seconds": run["configs"][config]["seconds"],
            }))

    seeds = []
    phobert_run = gpu_run("phobert", "dev")
    for seed in PHOBERT_SEEDS:
        predictions = read_gpu_predictions(phobert_run, f"dev_scores_{PHOBERT}_seed_{seed}.csv", dev)
        seeds.extend(metric_rows(dev, predictions, {"config": PHOBERT, "seed": seed}))
    seed_table = pd.DataFrame(seeds)
    seed_table.to_csv(RESULT / "phobert_seed_metrics.csv", index=False, encoding=CSV_ENCODING)
    seed_table.groupby("group")[["f1", "pr_auc"]].agg(["mean", "std"]).to_csv(
        RESULT / "phobert_seed_summary.csv", encoding=CSV_ENCODING,
    )
    full = pd.DataFrame(rows)
    full.to_csv(RESULT / "ladder_full.csv", index=False, encoding=CSV_ENCODING)
    return full


def final_labels(config: str) -> dict:
    hashes = data_hashes()
    dev, corpus = load_dev(TASK), load_corpus(TASK)
    mapping = dev_to_corpus_ids(TASK)
    overrides = dict(zip(mapping["corpus_id"], mapping["label"].astype(int)))
    duplicate_groups = int((mapping.groupby("id").size() > 1).sum())
    train, _ = public_training(dev)
    settings = {
        tfidf_name(source, input_name, decision): (source, input_name == "parent", decision)
        for source in SOURCES for input_name in INPUTS for decision in THRESHOLDS
    }
    details = {}
    start = time.perf_counter()
    if config in ("majority", "keywords"):
        score = baseline_scores(config, train, corpus)
        pred = (score >= 0.5).astype(int)
    elif config in settings:
        source, with_parent, decision = settings[config]
        fitted = fit_tfidf(train, dev, source, with_parent)
        score = tfidf_scores(fitted, corpus, with_parent)
        details = fitted[2]
        threshold = 0.5 if decision == "t05" else details["threshold_f1"]
        pred = (score >= threshold).astype(int)
        details["decision_threshold"] = threshold
    else:
        family = next((f for f, configs in GPU_CONFIGS.items() if config in configs), None)
        if family is None:
            raise ValueError(f"unknown model configuration {config!r}")
        run = gpu_run(family, "corpus")
        if config not in run["configs"]:
            raise ValueError(f"Kaggle predictions for {config!r} are missing")
        predictions = read_gpu_predictions(run, f"corpus_scores_{config}.csv", corpus)
        score, pred = predictions["score"].to_numpy(), predictions["pred"].to_numpy()
        details = {"gpu": run["gpu"], "gpu_seconds": run["configs"][config]["seconds"]}

    output = prediction_table(corpus, score, pred).rename(columns={"pred": "label"})
    manual = output["id"].map(overrides)
    in_dev = manual.notna().to_numpy()
    output["label"] = manual.fillna(output["label"]).astype(int)
    if data_hashes() != hashes:
        raise ValueError("task data changed during inference")
    path = DATA / "corpus_labels.csv"
    temp = path.with_suffix(".csv.tmp")
    output.to_csv(temp, index=False, encoding=CSV_ENCODING)
    temp.replace(path)
    random_dev = group_masks(dev)["random"]
    corpus_replies = corpus["parent_text"].str.strip().ne("").to_numpy()
    # The threshold is tuned on the oversampled dev; compare the model's positive rate on unseen
    # comments with the positive rate of the random dev sample, which estimates the corpus rate.
    summary = {
        **hashes, "config": config, "n": len(output), "dev_rows_overridden": int(in_dev.sum()),
        "dev_duplicate_groups": duplicate_groups, "raw_predicted_positive_share": round(float(pred.mean()), 4),
        "non_dev_predicted_positive_share": round(float(pred[~in_dev].mean()), 4),
        "dev_positive_share": round(float(labels(dev).mean()), 4),
        "dev_random_positive_share": round(float(labels(dev)[random_dev].mean()), 4),
        "final_positive_share": round(float(output["label"].mean()), 4),
        "final_positive_count": int(output["label"].sum()),
        "final_positive_by_type": {
            name: {"n": int(mask.sum()), "positives": int(output["label"][mask].sum())}
            for name, mask in (("root", ~corpus_replies), ("reply", corpus_replies))
        },
        "details": details, "local_seconds": round(time.perf_counter() - start, 3),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    write_json(RESULT / "final.json", summary)
    return summary


def find_scores(config: str) -> Path:
    for directory in (RESULT, KAGGLE):
        path = directory / f"dev_scores_{config}.csv"
        if path.exists():
            return path
    raise FileNotFoundError(f"dev_scores_{config}.csv not found in {RESULT} or {KAGGLE}")


def compare(config_a: str, config_b: str) -> pd.DataFrame:
    """Paired bootstrap of two configurations on the same dev items, overall and per group."""
    dev = load_dev(TASK)
    a = read_predictions(find_scores(config_a), dev)
    b = read_predictions(find_scores(config_b), dev)
    y = labels(dev)
    f1 = lambda truth, pred: f1_score(truth, pred, zero_division=0)
    rows = []
    for group, mask in group_masks(dev).items():
        if len(np.unique(y[mask])) < 2:
            continue
        d_f1 = paired_bootstrap(y[mask], a["pred"].to_numpy()[mask], b["pred"].to_numpy()[mask], f1)
        d_ap = paired_bootstrap(y[mask], a["score"].to_numpy()[mask], b["score"].to_numpy()[mask],
                                average_precision_score)
        rows.append({"group": group, "n": int(mask.sum()), "delta_f1": d_f1["delta"], "f1_ci95": d_f1["ci95"],
                     "delta_pr_auc": d_ap["delta"], "pr_auc_ci95": d_ap["ci95"]})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action")
    commands.add_parser("initial", help="evaluate cheap models trained without dev")
    ladder = commands.add_parser("ladder", help="evaluate the CPU ladder and all classical ablations")
    ladder.add_argument("--jobs", type=int, default=1, help="number of independent TF-IDF configurations to run in parallel")
    commands.add_parser("kaggle-bundle", help="prepare the incivility inputs for a private Kaggle dataset")
    commands.add_parser("collect", help="score the GPU outputs and join them to the CPU ladder")
    final = commands.add_parser("final", help="refit the selected model and write corpus_labels.csv")
    final.add_argument("--config", required=True, help="selected configuration from ladder_full.csv")
    comparison = commands.add_parser("compare", help="paired bootstrap of two configurations on dev (A minus B)")
    comparison.add_argument("config_a")
    comparison.add_argument("config_b")
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        return 2

    columns = ["config", "group", "n", "f1", "f1_ci95", "pr_auc", "pr_auc_ci95", "seconds"]
    if args.action in ("initial", "ladder"):
        jobs = getattr(args, "jobs", 1)
        if jobs < 1:
            parser.error("--jobs must be a positive integer")
        print(local_ladder(initial=args.action == "initial", jobs=jobs).to_string(
            columns=columns, index=False, float_format="%.4f",
        ))
    elif args.action == "kaggle-bundle":
        print(kaggle_bundle())
    elif args.action == "collect":
        print(collect().to_string(columns=columns, index=False, float_format="%.4f"))
    elif args.action == "final":
        print(json.dumps(final_labels(args.config), indent=2))
    elif args.action == "compare":
        print(compare(args.config_a, args.config_b).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
