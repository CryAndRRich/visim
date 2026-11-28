"""Generate simulated comment threads with an LLM through vLLM (run on a GPU server).

One thread per article and condition, with as many comments as the real thread. Comments are written in
chunks of at most CHUNK; each later chunk sees the comments written so far. Every chunk is checked with
`prompts.validate`; flagged comments (and the replies to them) are dropped, the rest is kept and renumbered,
and the next round writes the missing ones (with another seed). A thread with RETRIES attempts in a row that keep nothing is recorded as failed. All unfinished threads advance together, one chunk per round, so the GPU works
on a full batch. Finished threads are appended to <out>/threads.jsonl at once: a run that stops resumes
where it was (threads already in the file are skipped).

    python -m src.simulation.generate download                       # model into the Hugging Face cache
    python -m src.simulation.generate run --out data/simulation/generated/pilot \\
        --ids 5129869 5127176 5128283 5125819 5127763                # pilot: 5 articles x A, B, C
    python -m src.simulation.generate run --out data/simulation/generated/main     # all 100 articles
    python -m src.simulation.generate run --out data/simulation/generated/rewrite --conditions A C \\
        --titles data/simulation/rewritten_titles.csv                # headline rewrite: article_id,title
    python -m src.simulation.generate run --out /tmp/x --dry-run     # threads, chunks, prompt sizes; no model
    python -m src.simulation.generate summary --out data/simulation/generated/pilot [--show ID]
                                     # statistics by condition next to the real threads; no model

Output lines: article_id, condition, title (the rewritten headline, or null), n, comments [{id, parent, text}],
retries, failed (the problems of the last attempt, or null). <out>/run.json holds the settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..classifiers.common import CSV_ENCODING, SEED, run_info
from ..processing.snapshot import PROCESSED_DIR
from .prompts import (CHUNK, CONDITIONS, SCHEMA, SIM_DIR, article_inputs, messages, near_duplicate, strip_ids,
                      text_key, validate)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 6144          # one chunk: 25 comments
MAX_MODEL_LEN = 32768      # the longest thread (199 comments) sends about 150 earlier comments in its last chunk
GPU_MEMORY = 0.6           # share of the GPU vLLM may take (the server is shared)
RETRIES = 5               # attempts in a row without any accepted comment before a thread is given up
PROBLEM_AT = re.compile(r"comment (\d+):")

Complete = Callable[[list, list, list], list]   # (conversations, seeds, schemas) -> raw texts


@dataclass
class Thread:
    article: dict
    condition: str
    title: str | None = None
    comments: list = field(default_factory=list)
    retries: int = 0
    ids_stripped: int = 0
    attempt: int = 0            # attempts at the current chunk
    failed: list | None = None

    @property
    def key(self) -> tuple:
        return self.article["article_id"], self.condition

    @property
    def finished(self) -> bool:
        return self.failed is not None or len(self.comments) >= self.article["n"]

    def record(self) -> dict:
        return {"article_id": self.article["article_id"], "condition": self.condition, "title": self.title,
                "n": self.article["n"], "comments": self.comments, "retries": self.retries,
                "ids_stripped": self.ids_stripped, "failed": self.failed}


def chunk_seed(thread: Thread, start: int) -> int:
    text = f"{SEED}|{thread.article['article_id']}|{thread.condition}|{thread.title}|{start}|{thread.attempt}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def chunk_schema(start: int, count: int) -> dict:
    """The output schema with the exact number of comments of this chunk."""
    items = SCHEMA["properties"]["comments"]
    return {**SCHEMA, "properties": {"comments": {**items, "minItems": count, "maxItems": count}}}


def keep_valid(comments: list, problems: list[str], start: int, count: int) -> list[dict]:
    """The comments of a chunk without the flagged ones and the replies to them, renumbered from `start` with
    their parents remapped. One repeated comment then costs one comment, not the rest of the chunk, and the
    next round writes the missing ones."""
    if "invalid json" in problems:
        return []
    bad = {int(m.group(1)) for m in map(PROBLEM_AT.match, problems) if m}
    kept, new_id, dropped = [], {}, set()
    for i, c in enumerate(comments[:count]):
        if i in bad or not isinstance(c, dict) or c.get("parent") in dropped:
            if isinstance(c, dict):
                dropped.add(c.get("id"))
            continue
        nid = f"c{start + len(kept)}"
        new_id[c["id"]] = nid
        kept.append({"id": nid, "parent": new_id.get(c["parent"], c["parent"]), "text": c["text"]})
    return kept


def run_rounds(threads: list[Thread], complete: Complete, on_finish: Callable[[Thread], None] = lambda t: None,
               log: Callable[[str], None] = print, reasons: Counter | None = None) -> None:
    """Advance every unfinished thread by one chunk per round until all are finished."""
    round_no = 0
    reasons = Counter() if reasons is None else reasons
    while True:
        active = [t for t in threads if not t.finished]
        if not active:
            return
        round_no += 1
        jobs = []
        for t in active:
            start = len(t.comments) + 1
            count = min(CHUNK, t.article["n"] - start + 1)
            conv = messages(t.article, t.condition, start=start, count=count, previous=t.comments or None,
                            title=t.title)
            jobs.append((t, start, count, conv, chunk_seed(t, start)))
        began = time.time()
        raws = complete([j[3] for j in jobs], [j[4] for j in jobs], [chunk_schema(j[1], j[2]) for j in jobs])
        bad = 0
        for (t, start, count, _, _), raw in zip(jobs, raws):
            comments, problems = validate(raw, count, start, {c["id"] for c in t.comments},
                                          [c["text"] for c in t.comments])
            kept = keep_valid(comments, problems, start, count)
            for c in kept:
                text, stripped = strip_ids(str(c["text"]))
                t.ids_stripped += stripped
                t.comments.append({"id": c["id"], "parent": c["parent"], "text": text})
            for problem in problems:
                reasons[PROBLEM_AT.sub("", problem).strip()] += 1
            if problems:
                bad += 1
                t.retries += 1
            if kept:
                t.attempt = 0
            elif problems:
                t.attempt += 1
                if t.attempt > RETRIES:
                    t.failed = problems[:5]
            if t.finished:
                on_finish(t)
        done = sum(t.finished for t in threads)
        log(f"round {round_no}: {len(jobs)} chunks, {bad} cut or rejected, {time.time() - began:.0f} s; "
            f"{done}/{len(threads)} threads finished; problems so far {dict(reasons.most_common(4))}")


def prefer_ipv4() -> None:
    """Try IPv4 addresses first: on a host whose IPv6 has no route out, Python otherwise waits on IPv6."""
    original = socket.getaddrinfo

    def ordered(*args, **kwargs):
        return sorted(original(*args, **kwargs), key=lambda a: a[0] != socket.AF_INET)

    socket.getaddrinfo = ordered


def vllm_complete(model: str, gpu_memory: float, max_model_len: int, use_schema: bool,
                  repetition_penalty: float = 1.0) -> tuple[Complete, dict]:
    # FlashInfer's top-k/top-p sampler compiles its kernel at start-up and needs nvcc (a CUDA toolkit), which a
    # server with only the driver lacks; PyTorch's sampler needs nothing and costs little for a 7B model
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import vllm
    from vllm import LLM, SamplingParams

    try:
        from vllm.sampling_params import StructuredOutputsParams

        def constrain(schema):
            return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:   # older vLLM
        from vllm.sampling_params import GuidedDecodingParams

        def constrain(schema):
            return {"guided_decoding": GuidedDecodingParams(json=schema)}

    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_memory,
              seed=SEED)

    def complete(conversations, seeds, schemas):
        params = [SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, seed=s,
                                 repetition_penalty=repetition_penalty,
                                 **(constrain(sc) if use_schema else {}))
                  for s, sc in zip(seeds, schemas)]
        outputs = llm.chat(conversations, params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    return complete, {"vllm": vllm.__version__}


def build_threads(args) -> list[Thread]:
    arts = article_inputs()
    titles = {}
    if args.titles:
        t = pd.read_csv(args.titles, dtype={"article_id": str}, encoding=CSV_ENCODING)
        titles = dict(zip(t["article_id"], t["title"]))
        arts = [a for a in arts if a["article_id"] in titles]
    if args.ids:
        missing = set(args.ids) - {a["article_id"] for a in arts}
        if missing:
            raise SystemExit(f"not among the simulated articles: {sorted(missing)}")
        arts = [a for a in arts if a["article_id"] in set(args.ids)]
    if args.limit:
        arts = arts[:args.limit]
    return [Thread(a, c, titles.get(a["article_id"])) for a in arts for c in args.conditions]


def run(args) -> int:
    threads = build_threads(args)
    out = Path(args.out)
    path = out / "threads.jsonl"
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            done = {(r["article_id"], r["condition"]) for r in map(json.loads, f)}
    todo = [t for t in threads if t.key not in done]
    chunks = sum(-(-t.article["n"] // CHUNK) for t in todo)
    print(f"{len(threads)} threads ({len(done)} already in {path}); to do {len(todo)} threads, "
          f"{sum(t.article['n'] for t in todo)} comments, {chunks} chunks", flush=True)
    if args.dry_run:
        for t in todo[:3]:
            words = len(messages(t.article, t.condition, title=t.title)[1]["content"].split())
            print(f"  {t.key}: n={t.article['n']}, first prompt {words} words")
        return 0
    if not todo:
        return 0
    prefer_ipv4()
    began = time.time()
    complete, versions = vllm_complete(args.model, args.gpu_memory, args.max_model_len, not args.no_schema,
                                      args.repetition_penalty)
    out.mkdir(parents=True, exist_ok=True)
    settings = {**run_info(), **versions, "model": args.model, "temperature": TEMPERATURE, "top_p": TOP_P,
                "max_tokens": MAX_TOKENS, "repetition_penalty": args.repetition_penalty, "chunk": CHUNK, "retries": RETRIES, "schema": not args.no_schema,
                "conditions": args.conditions, "titles": args.titles, "threads": len(threads)}
    (out / "run.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(path, "a", encoding="utf-8") as f:
        def save(t: Thread) -> None:
            f.write(json.dumps(t.record(), ensure_ascii=False) + "\n")
            f.flush()
        reasons = Counter()
        run_rounds(todo, complete, on_finish=save, log=lambda m: print(m, flush=True), reasons=reasons)

    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    settings.update({"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - began) / 60, 1),
                     "failed": sum(r["failed"] is not None for r in records),
                     "retries": sum(r["retries"] for r in records), "records": len(records),
                     "ids_stripped": sum(r.get("ids_stripped", 0) for r in records),
                     "problems_this_run": dict(reasons.most_common())})
    (out / "run.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: settings[k] for k in ("records", "failed", "retries", "minutes")}), flush=True)
    return 0


def max_depth(comments: list[dict]) -> int:
    """Depth of the deepest comment; top-level comments have depth 1."""
    parent, best = {c["id"]: c["parent"] for c in comments}, 0
    for c in comments:
        d, p = 1, c["parent"]
        while p:
            d, p = d + 1, parent.get(p)
        best = max(best, d)
    return best


def thread_stats(texts: list[str], parents: list, depth: int) -> dict:
    words = [len(str(x).split()) for x in texts]
    keys = [text_key(x) for x in texts]
    repeats = sum(any(near_duplicate(k, e) for e in keys[:i]) for i, k in enumerate(keys))
    return {"n": len(texts), "reply_ratio": sum(p is not None for p in parents) / len(texts), "max_depth": depth,
            "words_mean": sum(words) / len(words), "words_median": float(pd.Series(words).median()),
            "words_max": max(words), "short_le5": sum(w <= 5 for w in words) / len(words), "repeats": repeats}


def summary(out: str, show: str | None = None) -> int:
    """Mean per thread of simple statistics, by condition and for the real threads of the same articles."""
    with open(Path(out) / "threads.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    ok = [r for r in records if r["failed"] is None]
    rows = [{"condition": r["condition"], **thread_stats([c["text"] for c in r["comments"]],
                                                         [c["parent"] for c in r["comments"]],
                                                         max_depth(r["comments"]))} for r in ok]
    ids = {r["article_id"] for r in ok}
    comments = pd.read_parquet(PROCESSED_DIR / "comments.parquet", columns=["article_id", "comment_id", "text"])
    comments["article_id"] = comments["article_id"].astype(str)
    threads = pd.read_parquet(PROCESSED_DIR / "threads.parquet", columns=["article_id", "reply_ratio", "max_depth_rec"])
    threads["article_id"] = threads["article_id"].astype(str)
    for aid, g in comments[comments["article_id"].isin(ids)].groupby("article_id"):
        th = threads[threads["article_id"] == aid].iloc[0]
        st = thread_stats(g["text"].tolist(), [None] * len(g), int(th["max_depth_rec"]))
        rows.append({"condition": "real", **st, "reply_ratio": float(th["reply_ratio"])})
    table = pd.DataFrame(rows).groupby("condition").mean().round(2)
    print(f"{len(records)} threads, {len(records) - len(ok)} failed, {sum(r['retries'] for r in records)} retries")
    print(table.to_string())
    ex = pd.read_csv(SIM_DIR / "examples.csv", dtype={"comment_id": str}, encoding=CSV_ENCODING)
    ex_keys = [text_key(x) for x in comments.assign(comment_id=comments["comment_id"].astype(str))
               .merge(ex, on="comment_id")["text"]]
    copied = sum(any(near_duplicate(text_key(c["text"]), k) for k in ex_keys)
                 for r in ok if r["condition"] == "B" for c in r["comments"])
    print(f"condition B comments repeating a style example: {copied}")
    for r in ok:
        if r["article_id"] == show:
            print(f"\n{show} condition {r['condition']}:")
            for c in r["comments"][:8]:
                print(f"  [{c['id']} <- {c['parent']}] {c['text'][:200]}")
    return 0


def download(model: str) -> int:
    prefer_ipv4()
    from huggingface_hub import snapshot_download
    print(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]), flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate simulated comment threads with vLLM")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download", help="download the model into the Hugging Face cache")
    d.add_argument("--model", default=MODEL)
    r = sub.add_parser("run", help="generate threads")
    r.add_argument("--out", required=True)
    r.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    r.add_argument("--ids", nargs="+", help="only these article ids")
    r.add_argument("--limit", type=int, help="only the first N articles")
    r.add_argument("--titles", help="CSV article_id,title: generate with these headlines (rewrite scenario)")
    r.add_argument("--model", default=MODEL)
    r.add_argument("--gpu-memory", type=float, default=GPU_MEMORY)
    r.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    r.add_argument("--no-schema", action="store_true", help="no constrained JSON decoding")
    r.add_argument("--repetition-penalty", type=float, default=1.0)
    r.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("summary", help="per-condition thread statistics of a run next to the real threads")
    s.add_argument("--out", required=True)
    s.add_argument("--show", help="print a few comments of this article id")
    a = p.parse_args(argv)
    if a.command == "summary":
        return summary(a.out, a.show)
    return download(a.model) if a.command == "download" else run(a)


if __name__ == "__main__":
    sys.exit(main())
