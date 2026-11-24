"""Agreement between the dev labels and an independent human annotator.

dev labels were assigned and reviewed with the help of an LLM, so every dev score measures agreement
with those labels. To know how far they are from a human reader, one team member labels a random sample
of dev without seeing the dev label, and the two are compared.

    python -m src.classifiers.agreement sample <task> [--n 100]
        writes data/tasks/<task>/agreement.csv (id, content columns, empty label): a random sample of dev,
        stratified by dev label, seed 42. It refuses to overwrite a file that already has labels.
    (fill the `label` column by hand, without AI tools and without looking at dev.csv)
    python -m src.classifiers.agreement score <task>
        compares the filled labels with dev.csv: raw agreement, Cohen's kappa, per-class counts;
        writes results/<task>/agreement.json and prints the same numbers.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

from .common import CONTENT, CSV_ENCODING, LABEL_VALUES, RESULTS_DIR, SEED, load_dev, run_info, task_dir


def path(task: str):
    return task_dir(task) / "agreement.csv"


def sample(task: str, n: int = 100) -> pd.DataFrame:
    out = path(task)
    if out.exists():
        old = pd.read_csv(out, dtype=str, encoding=CSV_ENCODING, keep_default_na=False)
        if (old["label"] != "").any():
            raise FileExistsError(f"{out} already has labels; delete it by hand to draw a new sample")
    dev = load_dev(task)
    picked, _ = train_test_split(dev, train_size=n, stratify=dev["label"], random_state=SEED)
    sheet = picked.sort_values("id")[["id", *CONTENT[task]]].assign(label="")
    sheet.to_csv(out, index=False, encoding=CSV_ENCODING)
    return sheet


def score(task: str) -> dict:
    sheet = pd.read_csv(path(task), dtype=str, encoding=CSV_ENCODING, keep_default_na=False)
    sheet["label"] = sheet["label"].str.strip()
    bad = sheet.loc[~sheet["label"].isin(LABEL_VALUES[task]), "id"]
    if len(bad):
        raise ValueError(f"{len(bad)} rows without a valid label ({LABEL_VALUES[task]}), e.g. ids {list(bad[:5])}")
    m = sheet[["id", "label"]].merge(load_dev(task)[["id", "label"]], on="id", suffixes=("_human", "_dev"))
    labels = LABEL_VALUES[task]
    result = {
        "task": task, "n": int(len(m)),
        "agreement": round(float((m["label_human"] == m["label_dev"]).mean()), 3),
        "cohen_kappa": round(float(cohen_kappa_score(m["label_human"], m["label_dev"], labels=labels)), 3),
        "labels": labels,
        "confusion_human_rows_dev_cols": confusion_matrix(m["label_human"], m["label_dev"], labels=labels).tolist(),
        "disagreeing_ids": sorted(m.loc[m["label_human"] != m["label_dev"], "id"]),
        **run_info(),
    }
    out = RESULTS_DIR / task
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Agreement between dev labels and a human annotator")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="draw the sample to label")
    s.add_argument("task", choices=list(CONTENT))
    s.add_argument("--n", type=int, default=100)
    c = sub.add_parser("score", help="compare the filled sample with dev.csv")
    c.add_argument("task", choices=list(CONTENT))
    a = p.parse_args(argv)
    if a.command == "sample":
        try:
            sheet = sample(a.task, a.n)
        except FileExistsError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"wrote {path(a.task)} ({len(sheet)} rows); fill the label column by hand")
    else:
        print(json.dumps(score(a.task), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
