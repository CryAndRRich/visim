"""Download the public datasets used for training (research use only; never committed to git).

ViClickbait-2025 (Mendeley Data, DOI 10.17632/3wc46bfcjc.1, CC BY 4.0): only the CSV is extracted.
The published dataset.zip is 1.16 GB because it also holds 3,410 thumbnail images. This script reads
the zip's central directory with HTTP range requests and extracts only
dataset/clickbait_dataset_vietnamese.csv (about 1.8 MB), in four requests.

UIT-ViCTSD (Nguyen, Nguyen and Nguyen, IEA/AIE 2021): the three CSV files released by the first author
on Hugging Face (tarudesu/ViCTSD; card: "The provided dataset is only used for research purposes!").

Usage (from the project root):
    python scripts/fetch_data.py
Output: data/external/viclickbait2025/clickbait_dataset_vietnamese.csv, data/external/victsd/ViCTSD_*.csv
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

LIST_URL = "https://data.mendeley.com/public-api/datasets/3wc46bfcjc/files?folder_id=root&version=1"
MEMBER = "dataset/clickbait_dataset_vietnamese.csv"
EXPECTED_SHA256 = "b6982b51eaea2fe9d486ad9748267e656fa00213a9bf659d4ed069e06fe88e03"
EXTERNAL = Path(__file__).resolve().parents[1] / "data" / "external"
OUT = EXTERNAL / "viclickbait2025" / "clickbait_dataset_vietnamese.csv"
VICTSD_URL = "https://huggingface.co/datasets/tarudesu/ViCTSD/resolve/main/{}"
VICTSD_FILES = ["ViCTSD_train.csv", "ViCTSD_valid.csv", "ViCTSD_test.csv"]
UA = {"User-Agent": "research-script (IT4868 course project)"}


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = dict(UA)
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    time.sleep(1.5)  # polite: one request at a time, spaced out
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        return r.read()


def _central_directory(url: str, size: int) -> list:
    tail = _get(url, size - 70000, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    n, cd_size, cd_off = struct.unpack("<H", tail[i + 10:i + 12])[0], *struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or n == 0xFFFF:  # zip64
        j = tail.rfind(b"PK\x06\x06")
        n, cd_size, cd_off = struct.unpack("<QQQ", tail[j + 32:j + 56])
    cd = _get(url, cd_off, cd_off + cd_size - 1)
    entries, p = [], 0
    while cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        offset, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra, q = cd[p + 46 + nlen:p + 46 + nlen + elen], 0
        while q < len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals, k = struct.unpack("<" + "Q" * (hlen // 8), extra[q + 4:q + 4 + hlen]), 0
                if usize == 0xFFFFFFFF:
                    usize, k = vals[k], k + 1
                if csize == 0xFFFFFFFF:
                    csize, k = vals[k], k + 1
                if offset == 0xFFFFFFFF:
                    offset = vals[k]
            q += 4 + hlen
        entries.append({"name": name, "method": method, "csize": csize, "offset": offset})
        p += 46 + nlen + elen + clen
    return entries


def fetch_victsd() -> None:
    out = EXTERNAL / "victsd"
    out.mkdir(parents=True, exist_ok=True)
    for name in VICTSD_FILES:
        path = out / name
        if path.exists():
            print(f"{path} already exists")
            continue
        path.write_bytes(_get(VICTSD_URL.format(name)))
        print(f"wrote {path}")


def fetch_viclickbait() -> int:
    if OUT.exists():
        print(f"{OUT} already exists")
        return 0
    info = json.loads(_get(LIST_URL))[0]
    url, size = info["content_details"]["download_url"], info["size"]
    entry = next(e for e in _central_directory(url, size) if e["name"] == MEMBER)
    head = _get(url, entry["offset"], entry["offset"] + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = entry["offset"] + 30 + nlen + elen
    raw = _get(url, start, start + entry["csize"] - 1)
    data = raw if entry["method"] == 0 else zlib.decompress(raw, -15)
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        print(f"sha256 mismatch: {digest}", file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data)} bytes)")
    return 0


def main() -> int:
    status = fetch_viclickbait()
    fetch_victsd()
    return status


if __name__ == "__main__":
    sys.exit(main())
