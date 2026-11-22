"""Evaluate incivility models and prepare blind reviews of the VnExpress dev labels.

Run from the repository root with `python -m src.classifiers.incivility --help`.
Model predictions are written to results/incivility. A completed review sheet
is required to change dev.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import shutil
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

from .common import (
    CSV_ENCODING, RESULTS_DIR, ROOT, SEED, binary_metrics, bootstrap_ci,
    load_corpus, load_dev, load_train, run_info, task_dir, write_results,
)

TASK = "incivility"
DATA = task_dir(TASK)
RESULT = RESULTS_DIR / TASK
REVIEWS = RESULT / "reviews"
KAGGLE = RESULT / "kaggle"
BUNDLE = ROOT / "data" / "kaggle" / TASK
FOLDS = 5
C_VALUES = (0.25, 1.0, 4.0)
SOURCES = ("victsd", "dev", "victsd_plus_dev")
INPUTS = ("comment", "parent")
THRESHOLDS = ("t05", "tf1")
GROUPS = ("all", "root", "reply")
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


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)[:-1]
    best = thresholds[np.isclose(f1, f1.max(), rtol=0, atol=1e-12)]
    return float(min(best, key=lambda t: (abs(t - 0.5), t)))


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
    threshold = best_f1_threshold(y, out_of_fold[best])
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
    return pd.DataFrame({"id": frame["id"].to_numpy(), "pred": pred, "score": score})


def metric_rows(dev: pd.DataFrame, predictions: pd.DataFrame, details: dict) -> list[dict]:
    y = labels(dev)
    score = predictions["score"].to_numpy()
    pred = predictions["pred"].to_numpy()
    replies = dev["parent_text"].str.strip().ne("").to_numpy()
    rows = []
    for group, mask in zip(GROUPS, (np.ones(len(dev), dtype=bool), ~replies, replies)):
        if not mask.any():
            continue
        group_y, group_s, group_p = y[mask], score[mask], pred[mask]
        metrics = binary_metrics(group_y, group_s, group_p)
        if len(np.unique(group_y)) == 2:
            metrics.update(bootstrap_ci(group_y, group_s, group_p))
        rows.append({**details, "group": group, "n": int(mask.sum()), "positives": int(group_y.sum()), **metrics})
    return rows


def tfidf_name(source: str, input_name: str, threshold: str) -> str:
    return f"tfidf_lr_{source}_{input_name}_{threshold}"


def timed_tfidf(train: pd.DataFrame, dev: pd.DataFrame, source: str, input_name: str):
    start = time.perf_counter()
    result = evaluate_tfidf(train, dev, source, input_name == "parent")
    return result, round(time.perf_counter() - start, 3)


def data_hashes() -> dict[str, str]:
    return {f"{name}_sha256": digest(DATA / f"{name}.csv") for name in INPUT_FILES}


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
        shutil.copyfile(DATA / f"{name}.csv", BUNDLE / f"incivility_{name}.csv")
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


def dev_labels_by_corpus_id(dev: pd.DataFrame, corpus: pd.DataFrame) -> tuple[dict[str, str], int]:
    """Resolve the dev aliases by exact comment and parent text within the frozen corpus."""
    by_id = {}
    by_content = defaultdict(list)
    for row in corpus.itertuples(index=False):
        content = (row.text, row.parent_text)
        by_id[row.id] = content
        by_content[content].append(row.id)

    overrides = {}
    duplicate_groups = 0
    for row in dev.itertuples(index=False):
        content = (row.text, row.parent_text)
        if row.id in by_id:
            if by_id[row.id] != content:
                raise ValueError(f"dev and corpus text disagree for ID {row.id}")
            matches = [row.id]
        else:
            matches = by_content[content]
        if not matches:
            raise ValueError(f"no exact corpus match for dev ID {row.id}")
        duplicate_groups += len(matches) > 1
        for corpus_id in matches:
            if overrides.setdefault(corpus_id, row.label) != row.label:
                raise ValueError(f"conflicting dev labels for corpus ID {corpus_id}")
    return overrides, duplicate_groups


def final_labels(config: str) -> Path:
    hashes = data_hashes()
    dev, corpus = load_dev(TASK), load_corpus(TASK)
    overrides, duplicate_groups = dev_labels_by_corpus_id(dev, corpus)
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
    output["label"] = manual.fillna(output["label"]).astype(int)
    if data_hashes() != hashes:
        raise ValueError("task data changed during inference")
    path = DATA / "corpus_labels.csv"
    temp = path.with_suffix(".csv.tmp")
    output.to_csv(temp, index=False, encoding=CSV_ENCODING)
    temp.replace(path)
    RESULT.mkdir(parents=True, exist_ok=True)
    write_json(RESULT / "final.json", {
        **hashes, "config": config, "n": len(output), "dev_rows_overridden": len(overrides),
        "dev_duplicate_groups": duplicate_groups, "raw_predicted_positive_share": float(pred.mean()),
        "final_positive_share": float(output["label"].mean()), "final_positive_count": int(output["label"].sum()),
        "details": details, "local_seconds": round(time.perf_counter() - start, 3),
    })
    return path


def review_paths(round_number: int) -> tuple[Path, Path, Path]:
    if round_number < 1:
        raise ValueError("review round numbers start at 1")
    stem = f"round_{round_number:02d}"
    return tuple(REVIEWS / f"{stem}{suffix}" for suffix in (".csv", "_selection.json", "_audit.json"))


def prepare_review(round_number: int, initial: bool) -> Path:
    dev = load_dev(TASK)
    dev_sha = digest(DATA / "dev.csv")
    directory = RESULT / "initial" if initial else RESULT
    report = read_json(directory / "metrics.json")
    if report["dev_sha256"] != dev_sha:
        raise ValueError("rerun the models on the current dev.csv before preparing a review")
    candidates = [row for row in report["results"] if row["group"] == "all"]
    best = max(candidates, key=lambda row: (row["f1"], row["pr_auc"]))
    path, selection_path, audit_path = review_paths(round_number)
    if any(p.exists() for p in (path, selection_path, audit_path)):
        raise FileExistsError(f"review round {round_number} already exists")

    reviewed = set()
    for earlier in range(1, round_number):
        reviewed.update(read_json(review_paths(earlier)[2])["reviewed_ids"])
    scores = read_predictions(directory / f"dev_scores_{best['config']}.csv", dev).set_index("id")["score"]
    available = scores.drop(list(reviewed), errors="ignore")
    if len(available) < 30:
        raise ValueError("fewer than 30 unreviewed dev comments remain")
    # Match the dev prevalence for the sampling cutoff only. Reported predictions stay fixed.
    cutoff = float(available.quantile(1 - labels(dev).mean()))
    positive = available.nlargest(10)
    negative = available.drop(positive.index).nsmallest(10)
    uncertain = (available.drop(positive.index.union(negative.index)) - cutoff).abs().nsmallest(10)
    groups = {"likely_1": list(positive.index), "likely_0": list(negative.index), "uncertain": list(uncertain.index)}
    sample = dev.set_index("id").loc[list(positive.index) + list(negative.index) + list(uncertain.index)]
    sample = sample[["text", "parent_text"]].sample(frac=1, random_state=SEED + round_number - 1)
    sample = sample.assign(label="", reason="")
    REVIEWS.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, encoding=CSV_ENCODING)
    write_json(selection_path, {
        "round": round_number, "dev_sha256": dev_sha, "model": best["config"],
        "sampling_cutoff": cutoff, "groups": groups, "initial": initial,
    })
    return path


def update_dev_labels(path: Path, replacements: dict[str, str]) -> None:
    """Change only the final label digit of the specified CSV records."""
    raw = path.read_bytes().decode("utf-8")
    lines = io.StringIO(raw, newline="").readlines()
    reader = csv.reader(lines)
    header = next(reader)
    if header != ["\ufeffid", "text", "parent_text", "label"]:
        raise ValueError("expected the original dev.csv UTF-8 BOM and column order")
    pending = set(replacements)
    for row in reader:
        if row[0] in pending:
            index = reader.line_num - 1
            updated, count = re.subn(r",[01](?=\r?\n$|\r$|$)", "," + replacements[row[0]], lines[index])
            if count != 1:
                raise ValueError(f"expected a single final label digit for ID {row[0]}")
            lines[index] = updated
            pending.remove(row[0])
    if pending:
        raise ValueError(f"reviewed IDs are missing from dev.csv: {sorted(pending)}")
    temp = path.with_suffix(".csv.tmp")
    temp.write_bytes("".join(lines).encode("utf-8"))
    temp.replace(path)


def apply_review(round_number: int, reviewed_path: Path | None) -> Path:
    sheet, selection_path, audit_path = review_paths(round_number)
    if audit_path.exists():
        raise FileExistsError(f"review round {round_number} has already been applied")
    selection = read_json(selection_path)
    path = DATA / "dev.csv"
    before = digest(path)
    if selection["dev_sha256"] != before:
        raise ValueError("dev.csv changed since this review was prepared")
    reviewed = pd.read_csv(reviewed_path or sheet, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)
    expected = {item for group in selection["groups"].values() for item in group}
    if reviewed["id"].duplicated().any() or set(reviewed["id"]) != expected:
        raise ValueError("the completed sheet must cover exactly the 30 selected IDs")
    if not reviewed["label"].isin(["0", "1"]).all():
        raise ValueError("enter a reviewed label of 0 or 1 for every selected comment")

    current = load_dev(TASK).set_index("id").loc[reviewed["id"]].reset_index()
    for field in ("text", "parent_text"):
        if field in reviewed and not current[field].equals(reviewed[field]):
            raise ValueError(f"the review sheet changed {field}; edit only label and reason")
    old = current.set_index("id")["label"]
    changes = [
        {"id": row.id, "from": old[row.id], "to": row.label, "reason": getattr(row, "reason", "")}
        for row in reviewed.itertuples(index=False) if old[row.id] != row.label
    ]
    if changes:
        update_dev_labels(path, {row["id"]: row["to"] for row in changes})
    write_json(audit_path, {
        **selection, "dev_sha256_after": digest(path), "reviewed_ids": list(reviewed["id"]), "changes": changes,
        "reviewer_notes": {
            row.id: row.reason for row in reviewed.itertuples(index=False) if getattr(row, "reason", "")
        },
    })
    return audit_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("initial", help="evaluate cheap models trained without dev")
    ladder = commands.add_parser("ladder", help="evaluate the CPU ladder and all classical ablations")
    ladder.add_argument("--jobs", type=int, default=1, help="number of independent TF-IDF configurations to run in parallel")
    review = commands.add_parser("review", help="prepare a blind dev review sheet")
    review.add_argument("--round", type=int, required=True)
    review.add_argument("--full-ladder", action="store_true", help="select from ladder instead of initial")
    apply = commands.add_parser("apply-review", help="apply a completed review sheet")
    apply.add_argument("--round", type=int, required=True)
    apply.add_argument("--file", type=Path, help="completed sheet, if not edited in place")
    commands.add_parser("kaggle-bundle", help="prepare the incivility inputs for a private Kaggle dataset")
    commands.add_parser("collect", help="score the GPU outputs and join them to the CPU ladder")
    final = commands.add_parser("final", help="refit the selected model and write corpus_labels.csv")
    final.add_argument("--config", required=True, help="selected configuration from ladder_full.csv")
    args = parser.parse_args(argv)

    columns = ["config", "group", "n", "f1", "f1_ci95", "pr_auc", "pr_auc_ci95", "seconds"]
    if args.action in ("initial", "ladder"):
        jobs = getattr(args, "jobs", 1)
        if jobs < 1:
            parser.error("--jobs must be a positive integer")
        print(local_ladder(initial=args.action == "initial", jobs=jobs).to_string(
            columns=columns, index=False, float_format="%.4f",
        ))
    elif args.action == "review":
        print(prepare_review(args.round, initial=not args.full_ladder))
    elif args.action == "apply-review":
        print(apply_review(args.round, args.file))
    elif args.action == "kaggle-bundle":
        print(kaggle_bundle())
    elif args.action == "collect":
        print(collect().to_string(columns=columns, index=False, float_format="%.4f"))
    elif args.action == "final":
        print(final_labels(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
