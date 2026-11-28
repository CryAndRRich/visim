"""Generate simulated comment threads with an LLM through vLLM (run on a GPU server).

One thread per article and condition, with as many comments as the real thread. Comments are written in
chunks of at most CHUNK; each later chunk sees the comments written so far. Every chunk is checked with
`prompts.validate`; a failed chunk is sampled again with another seed, up to RETRIES times, after which the
thread is recorded as failed. All unfinished threads advance together, one chunk per round, so the GPU works
on a full batch. Finished threads are appended to <out>/threads.jsonl at once: a run that stops resumes
where it was (threads already in the file are skipped).

    python -m src.simulation.generate download                       # model into the Hugging Face cache
    python -m src.simulation.generate run --out data/simulation/generated/pilot \\
        --ids 5129869 5127176 5128283 5125819 5127763                # pilot: 5 articles x A, B, C
    python -m src.simulation.generate run --out data/simulation/generated/main     # all 100 articles
    python -m src.simulation.generate run --out data/simulation/generated/rewrite --conditions A C \\
        --titles data/simulation/rewritten_titles.csv                # headline rewrite: article_id,title
    python -m src.simulation.generate run --out /tmp/x --dry-run     # threads, chunks, prompt sizes; no model

Output lines: article_id, condition, title (the rewritten headline, or null), n, comments [{id, parent, text}],
retries, failed (the problems of the last attempt, or null). <out>/run.json holds the settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..classifiers.common import CSV_ENCODING, SEED, run_info
from .prompts import CHUNK, CONDITIONS, SCHEMA, article_inputs, messages, validate

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 6144          # one chunk: 25 comments
MAX_MODEL_LEN = 32768      # the longest thread (199 comments) sends about 150 earlier comments in its last chunk
GPU_MEMORY = 0.6           # share of the GPU vLLM may take (the server is shared)
RETRIES = 3

Complete = Callable[[list, list, list], list]   # (conversations, seeds, schemas) -> raw texts


@dataclass
class Thread:
    article: dict
    condition: str
    title: str | None = None
    comments: list = field(default_factory=list)
    retries: int = 0
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
                "n": self.article["n"], "comments": self.comments, "retries": self.retries, "failed": self.failed}


def chunk_seed(thread: Thread, start: int) -> int:
    text = f"{SEED}|{thread.article['article_id']}|{thread.condition}|{thread.title}|{start}|{thread.attempt}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def chunk_schema(start: int, count: int) -> dict:
    """The output schema with the exact number of comments of this chunk."""
    items = SCHEMA["properties"]["comments"]
    return {**SCHEMA, "properties": {"comments": {**items, "minItems": count, "maxItems": count}}}


def run_rounds(threads: list[Thread], complete: Complete, on_finish: Callable[[Thread], None] = lambda t: None,
               log: Callable[[str], None] = print) -> None:
    """Advance every unfinished thread by one chunk per round until all are finished."""
    round_no = 0
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
            comments, problems = validate(raw, count, start, {c["id"] for c in t.comments})
            if problems:
                bad += 1
                t.retries += 1
                t.attempt += 1
                if t.attempt > RETRIES:
                    t.failed = problems[:5]
            else:
                t.comments += [{"id": c["id"], "parent": c["parent"], "text": str(c["text"]).strip()}
                               for c in comments]
                t.attempt = 0
            if t.finished:
                on_finish(t)
        done = sum(t.finished for t in threads)
        log(f"round {round_no}: {len(jobs)} chunks, {bad} rejected, {time.time() - began:.0f} s; "
            f"{done}/{len(threads)} threads finished")


def prefer_ipv4() -> None:
    """Try IPv4 addresses first: on a host whose IPv6 has no route out, Python otherwise waits on IPv6."""
    original = socket.getaddrinfo

    def ordered(*args, **kwargs):
        return sorted(original(*args, **kwargs), key=lambda a: a[0] != socket.AF_INET)

    socket.getaddrinfo = ordered


def vllm_complete(model: str, gpu_memory: float, max_model_len: int, use_schema: bool) -> tuple[Complete, dict]:
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
    complete, versions = vllm_complete(args.model, args.gpu_memory, args.max_model_len, not args.no_schema)
    out.mkdir(parents=True, exist_ok=True)
    settings = {**run_info(), **versions, "model": args.model, "temperature": TEMPERATURE, "top_p": TOP_P,
                "max_tokens": MAX_TOKENS, "chunk": CHUNK, "retries": RETRIES, "schema": not args.no_schema,
                "conditions": args.conditions, "titles": args.titles, "threads": len(threads)}
    (out / "run.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(path, "a", encoding="utf-8") as f:
        def save(t: Thread) -> None:
            f.write(json.dumps(t.record(), ensure_ascii=False) + "\n")
            f.flush()
        run_rounds(todo, complete, on_finish=save, log=lambda m: print(m, flush=True))

    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    settings.update({"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - began) / 60, 1),
                     "failed": sum(r["failed"] is not None for r in records),
                     "retries": sum(r["retries"] for r in records), "records": len(records)})
    (out / "run.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: settings[k] for k in ("records", "failed", "retries", "minutes")}), flush=True)
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
    r.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    return download(a.model) if a.command == "download" else run(a)


if __name__ == "__main__":
    sys.exit(main())
