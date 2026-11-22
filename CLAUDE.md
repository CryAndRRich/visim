# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Course project (IT4868 Web Mining): "Real or Simulated Readers? Mining Vietnamese News Comments and Evaluating LLM-Generated Discussions". VnExpress articles and comments were crawled, anonymized and frozen (1,244 articles, 20,468 comments, snapshot of 2026-10-13); the current work is three classification tasks on that data. `TASK.md` (Vietnamese) is the authoritative spec for the classification work: task assignment, label definitions (section 6), evaluation rules, required deliverables and the notes format. Read it before changing anything under `src/classifiers/`, `notebooks/` or `data/tasks/`.

## Commands

Python 3.9, `pip install -r requirements.txt`. Every module uses package-relative imports, so always run from the project root as `python -m src.<pkg>.<module>`, never as a file path. On the project Mac the 3.9 environment with the pinned packages is `/usr/bin/python3`; the default `python3` on PATH is a 3.11 environment with different versions (pandas 3) and no pyarrow.

```bash
# Classifiers (clickbait.py is the template; incivility.py is still to be written in the same shape)
python -m src.classifiers.clickbait ladder         # baselines + TF-IDF models + ablations, ~10 min CPU
python -m src.classifiers.clickbait kaggle-bundle  # writes data/kaggle/clickbait/ for upload to Kaggle
python -m src.classifiers.clickbait collect        # merges results/clickbait/kaggle/dev_scores_*.csv into ladder_full.csv
python -m src.classifiers.sentiment ladder         # local levels, ~1.5 min; see its docstring for review /
                                                   # apply-review / collect / predict-corpus / label-corpus

# Rebuild data (normally not needed; everything is committed)
python scripts/fetch_data.py                       # public datasets -> data/external/
python -m src.classifiers.data                     # data/tasks/<task>/{train,corpus}.csv (never dev.csv)

# Graph / structure features (write into data/processed/)
python -m src.features.structure
python -m src.features.user_graph
python -m src.features.reply_graph
python -m src.processing.edge_check sample|show N|summary
```

`tests/` and `results/` are gitignored (kept local only). pytest is in requirements; run a single test with `python -m pytest tests/<file>.py::<test_name>` if the local `tests/` exists.

The crawler (`python -m src.crawler.run discover|comments|daily|stats`) and `python -m src.processing.snapshot` are historical: collection has stopped and `data/raw/` is no longer kept, so they document how `data/processed/` was made rather than being part of the normal workflow.

## Architecture

Data flows in one direction:

1. **Crawler** (`src/crawler/`): `PoliteSession` enforces robots.txt, a 1.5 s per-host interval, retries, and aborts the whole run on 403/429 (exit code 2). Comments come from an undocumented VnExpress API that only exposes two levels (top-level + replies). `Store` keeps crawl state in SQLite and content in JSONL. Comments are collected once, 48 h after publication.
2. **Anonymization** (`src/processing/anonymize.py`): user ids are HMAC-SHA256 hashed with the key in `secrets/hash_key` (gitignored; must never be committed or deleted, or hashes stop matching). Mentions become `@USER_<12 hex>`, phones/emails become `[PHONE]`/`[EMAIL]`.
3. **Snapshot** (`src/processing/snapshot.py`) → `data/processed/*.parquet` + `manifest.json`. `PROJECT_ROOT` and `PROCESSED_DIR` defined here are imported across the codebase.
4. **Reply trees** (`src/processing/trees.py`): the key non-obvious piece. `two_level_parents` uses the API parent (depth ≤ 2). `reconstructed_parents` re-attaches a reply that mentions a user to that user's latest earlier comment in the same top-level thread, recovering deeper conversations. `ROOT = 0` is the virtual article node. Both `features/` and `classifiers/data.py` (to fill `parent_text`) depend on the reconstructed version.
5. **Features** (`src/features/`): tree metrics (depth, structural virality), user–article bipartite projection with Louvain, and a directed user reply graph (PageRank, HITS, community method comparison). Docstrings note that "central/frequent user" sets must be recomputed on the training period only for controversy prediction, to avoid leakage.
6. **Classifiers** (`src/classifiers/`): `common.py` holds the shared loaders (`load_dev/train/corpus`), metrics, bootstrap CIs, `run_info()` and `write_results()`. Large models (PhoBERT, ViSoBERT fine-tuning, Qwen2.5) run on Kaggle GPU via `notebooks/<task>_<model>.ipynb`; the local machine's torch 2.2 can only load `.safetensors` checkpoints.

## Classification conventions (from TASK.md)

- Tasks and labels: `clickbait` (`0`/`1`, on `title`+`sapo`), `sentiment` (`neg`/`neu`/`pos`), `incivility` (`0`/`1`); comment tasks use `text`+`parent_text`. CSVs are read with `dtype=str`, `keep_default_na=False` and written with `utf-8-sig`, so labels are strings.
- `dev.csv` labels were assigned by an LLM and are being corrected by hand. For clickbait and incivility, labels must be decided by a human: do not decide or change their `dev.csv` labels yourself (TASK.md forbids AI tools for this). Sentiment is the one exception (TASK.md section 4): review rounds there may be labelled by Claude, with `labeled_by` and a short `reason` for every item in `notes/sentiment_review/round_N.csv`, and the notes must say the labels are LLM-reviewed, not human-checked. When editing `dev.csv`, change only the `label` column; one commit per review round. `dev.csv` is not representative (positives were oversampled), so don't estimate label rates from it.
- There is no test set. Models not trained on dev are scored on all of dev; models that train on dev use out-of-fold predictions from `StratifiedKFold(5, shuffle=True, random_state=42)` with hyperparameters chosen by nested CV. Thresholds/hyperparameters are never chosen on the reported part. Few-shot examples must not include the scored item (take them from `train.csv`; sentiment has none).
- Seed 42; stochastic models run 3 seeds (mean ± std).
- Per-config dev predictions go to `results/<task>/dev_scores_<config>.csv` with `id`, `pred`, and `score` (binary, P(class 1)) or `p_neg,p_neu,p_pos` (sentiment). Final corpus labels go to `data/tasks/<task>/corpus_labels.csv` (`id,label,score`, keeping hand labels for dev items). Write-ups go in `notes/<task>.md`; since `results/` is not in git, key numbers must be copied there.
- `dev.csv` ids are local (`001`, `E288`, ...), not corpus ids. To join dev with `corpus.csv`, match on the content columns (`sentiment.dev_to_corpus_ids()` does this for comments; a few corpus comments are exact duplicates).
- The pretrained `5CD-AI/Vietnamese-Sentiment-visobert` head is ordered `NEG, POS, NEU`; always map through `config.id2label`.
- Data on Kaggle goes only into private datasets. The clickbait bundle holds headline text only; comment tasks upload `data/tasks/<task>/` as TASK.md describes.
- Workflow: branch `<task>-<name>` off `curr`, PR back into `curr`.
