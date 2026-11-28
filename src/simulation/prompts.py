"""Prompts, output schema and output checks for the reader simulation (generation runs on a GPU server).

Conditions (one thread per article and condition, as many comments as the real thread):
    A   title, sapo and article body
    B   as A, plus N_EXAMPLES real comments of the same category from the earlier period (style examples)
    C   title only

The model answers with JSON: {"comments": [{"id": "c1", "parent": null, "text": "..."}, ...]}, where
`parent` is null for a top-level comment or the id of an earlier comment it replies to. Long threads are
generated in chunks of at most CHUNK comments; each later chunk gets the comments written so far.

    python -m src.simulation.prompts preview ID C    # writes the first-chunk prompt of article ID, condition
                                                     # C, to results/simulation/preview_<ID>_<C>.txt
`validate(raw, n)` checks one generated thread (JSON, number of comments, parent links, language).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import pandas as pd

from ..classifiers.common import CSV_ENCODING, RESULTS_DIR, ROOT
from ..processing.snapshot import PROCESSED_DIR

SIM_DIR = ROOT / "data" / "simulation"
CONDITIONS = ("A", "B", "C")
CHUNK = 25
MAX_BODY_WORDS = 1200
SCHEMA = {
    "type": "object", "required": ["comments"],
    "properties": {"comments": {"type": "array", "items": {
        "type": "object", "required": ["id", "parent", "text"],
        "properties": {"id": {"type": "string"}, "parent": {"type": ["string", "null"]}, "text": {"type": "string"}},
    }}},
}
SYSTEM = (
    "Bạn mô phỏng độc giả báo VnExpress viết bình luận dưới một bài báo. Viết bằng tiếng Việt, giống bình luận "
    "thật trên báo điện tử: giọng điệu và quan điểm khác nhau giữa các độc giả; có người đồng tình, có người phản "
    "đối, có người hỏi thêm; một số bình luận trả lời bình luận trước đó. Độ dài khác nhau: phần lớn một đến ba "
    "câu, một số chỉ vài từ, thỉnh thoảng có bình luận dài nhiều câu phân tích kỹ hoặc kể trải nghiệm riêng. Văn "
    "phong tự nhiên như người đọc thật, không trang trọng, không tóm tắt lại bài báo, không lặp lại ý hay câu chữ "
    "của bình luận đã có. Không nhắc id (c1, c2...) trong nội dung. Không nêu tên thật, số điện thoại hay email. "
    "Chỉ trả về JSON đúng schema, không viết gì thêm."
)
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
ID_IN_TEXT = re.compile(r"\bc\d{1,3}\b")   # lower case only: "cúp C1" is a football cup
WORD = re.compile(r"\w+")
NEAR_DUPLICATE = 0.8   # word-set Jaccard at or above which two comments count as the same


def text_key(text: str) -> tuple:
    return tuple(WORD.findall(str(text).lower()))


def near_duplicate(a: tuple, b: tuple) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 6:
        return False
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) >= NEAR_DUPLICATE


def article_inputs() -> list[dict]:
    """One record per simulated article with the text fields and the style examples of its category."""
    chosen = pd.read_csv(SIM_DIR / "articles.csv", dtype={"article_id": str}, encoding=CSV_ENCODING)
    arts = pd.read_parquet(PROCESSED_DIR / "articles.parquet", columns=["article_id", "title", "sapo", "body"])
    arts["article_id"] = arts["article_id"].astype(str)
    ex = pd.read_csv(SIM_DIR / "examples.csv", dtype={"comment_id": str}, encoding=CSV_ENCODING)
    com = pd.read_parquet(PROCESSED_DIR / "comments.parquet", columns=["comment_id", "text"])
    com["comment_id"] = com["comment_id"].astype(str)
    ex = ex.merge(com, on="comment_id")
    by_cat = ex.groupby("category")["text"].apply(list).to_dict()
    rows = chosen.merge(arts, on="article_id")
    return [{"article_id": r.article_id, "category": r.category, "clickbait": int(r.clickbait), "n": int(r.size),
             "title": r.title, "sapo": r.sapo or "", "body": " ".join(str(r.body or "").split()[:MAX_BODY_WORDS]),
             "examples": by_cat[r.category]} for r in rows.itertuples()]


def user_prompt(art: dict, condition: str, start: int = 1, count: int | None = None,
                previous: list[dict] | None = None, title: str | None = None) -> str:
    """User message for comments c{start}..c{start+count-1}. `title` replaces the headline (rewrite scenario)."""
    count = count or min(CHUNK, art["n"] - start + 1)
    head = title or art["title"]
    parts = [f"Tiêu đề: {head}"]
    if condition in ("A", "B"):
        parts += [f"Sapo: {art['sapo']}", f"Nội dung: {art['body']}"]
    if condition == "B":
        parts.append("Ví dụ bình luận thật của độc giả cùng chuyên mục (chỉ để tham khảo văn phong, không chép lại):\n"
                     + "\n".join(f"- {t}" for t in art["examples"]))
    if previous:
        parts.append("Các bình luận đã có:\n" + json.dumps({"comments": previous}, ensure_ascii=False))
    end = start + count - 1
    parts.append(
        f"Viết {count} bình luận mới, id từ c{start} đến c{end}, theo thứ tự thời gian. Trường parent là null nếu "
        f"bình luận viết thẳng về bài, hoặc là id của một bình luận trước đó (c1 đến c{end - 1}) nếu đó là câu trả "
        "lời. Trả về JSON dạng {\"comments\": [{\"id\": \"c" + str(start) + "\", \"parent\": null, \"text\": \"...\"}]}."
    )
    return "\n\n".join(parts)


def messages(art: dict, condition: str, **kw) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(art, condition, **kw)}]


def validate(raw: str, n: int, start: int = 1, known: set | None = None,
             earlier_texts: list | None = None) -> tuple[list[dict], list[str]]:
    """Parse one chunk; returns (comments, problems). Problems: bad JSON, wrong count or ids, parent not an
    earlier id, empty text, CJK characters (the model drifting out of Vietnamese), a comment id in the text,
    a comment that repeats
    (exactly or nearly) an earlier comment or another comment of the chunk."""
    problems = []
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        comments = json.loads(m.group(0))["comments"] if m else None
    except (json.JSONDecodeError, KeyError, TypeError):
        comments = None
    if not isinstance(comments, list):
        return [], ["invalid json"]
    if len(comments) != n:
        problems.append(f"{len(comments)} comments instead of {n}")
    seen = set(known or ())
    keys = [text_key(t) for t in earlier_texts or ()]
    for i, c in enumerate(comments):
        if not isinstance(c, dict) or not str(c.get("text", "")).strip():
            problems.append(f"comment {i}: empty or malformed")
            continue
        if c.get("id") != f"c{start + i}":
            problems.append(f"comment {i}: id {c.get('id')} instead of c{start + i}")
        if c.get("parent") is not None and c.get("parent") not in seen:
            problems.append(f"comment {i}: parent {c.get('parent')} is not an earlier comment")
        if CJK.search(str(c.get("text", ""))):
            problems.append(f"comment {i}: non-Vietnamese characters")
        if ID_IN_TEXT.search(str(c.get("text", ""))):
            problems.append(f"comment {i}: mentions a comment id")
        key = text_key(c.get("text", ""))
        if any(near_duplicate(key, k) for k in keys):
            problems.append(f"comment {i}: repeats an earlier comment")
        keys.append(key)
        seen.add(c.get("id"))
    return comments, problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reader simulation prompts")
    sub = p.add_subparsers(dest="command", required=True)
    pv = sub.add_parser("preview")
    pv.add_argument("article_id")
    pv.add_argument("condition", choices=CONDITIONS)
    a = p.parse_args(argv)
    art = next(x for x in article_inputs() if x["article_id"] == a.article_id)
    out = RESULTS_DIR / "simulation" / f"preview_{a.article_id}_{a.condition}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    msg = messages(art, a.condition)
    out.write_text(f"[system]\n{msg[0]['content']}\n\n[user]\n{msg[1]['content']}\n", encoding="utf-8")
    print(f"wrote {out} ({len(msg[1]['content'].split())} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
