#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import requests


def download(url, path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # THUAIPoet GitHub repository archive.
    thu_dir = root / "THUNLP-AIPoet-Datasets"
    if not thu_dir.exists():
        print("Downloading THUNLP-AIPoet/Datasets ...")
        url = "https://github.com/THUNLP-AIPoet/Datasets/archive/refs/heads/master.zip"
        r = requests.get(url, timeout=120)
        if r.status_code == 404:
            url = "https://github.com/THUNLP-AIPoet/Datasets/archive/refs/heads/main.zip"
            r = requests.get(url, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(root)
        candidates = sorted(root.glob("Datasets-*"))
        if candidates:
            candidates[0].rename(thu_dir)

    # Zenodo record: resolve current files through public API.
    zenodo_dir = root / "CCSC_Zenodo_17798065"
    zenodo_dir.mkdir(exist_ok=True)
    meta_url = "https://zenodo.org/api/records/17798065"
    print("Resolving CCSC Zenodo record ...")
    rec = requests.get(meta_url, timeout=60)
    rec.raise_for_status()
    metadata = rec.json()
    (zenodo_dir / "zenodo_record.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in metadata.get("files", []):
        name = f["key"]
        dst = zenodo_dir / name
        if dst.exists():
            continue
        link = f.get("links", {}).get("content") or f.get("links", {}).get("self")
        if link:
            print("Downloading", name)
            download(link, dst)

    print("Data root:", root.resolve())


if __name__ == "__main__":
    main()
