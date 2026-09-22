#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from preprocess_utils import (
    candidates_for, chars_from_lines, deterministic_split, labels_from_lexicon,
    load_lexicon, normalize_lines, write_splits,
)


def find_db(root: Path):
    dbs = list(root.rglob("*.sqlite")) + list(root.rglob("*.db")) + list(root.rglob("*.sqlite3"))
    if not dbs:
        raise FileNotFoundError("No SQLite database found under " + str(root))
    return dbs[0]


def choose_table(conn):
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    best = None
    for t in tables:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
        low = {c.lower(): c for c in cols}
        content = next((low[k] for k in ["content", "text", "poem", "full_text"] if k in low), None)
        cipai = next((low[k] for k in ["cipai", "rhythmic_title", "tune", "pattern"] if k in low), None)
        author = next((low[k] for k in ["author", "poet"] if k in low), None)
        if content and cipai:
            return t, content, cipai, author
        if content and best is None:
            best = (t, content, cipai, author)
    if best:
        return best
    raise RuntimeError("Could not identify a poem-content table. Inspect SQLite schema manually.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/CCSC_Zenodo_17798065")
    ap.add_argument("--out", default="data/processed_ccsc")
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--min-form-count", type=int, default=5)
    args = ap.parse_args()
    lexicon = load_lexicon(args.lexicon)
    db = find_db(Path(args.root))
    conn = sqlite3.connect(db)
    table, content_col, cipai_col, author_col = choose_table(conn)
    cols = [content_col, cipai_col] + ([author_col] if author_col else [])
    query = "SELECT " + ",".join(f'"{c}"' for c in cols) + f' FROM "{table}"'
    raw = list(conn.execute(query))
    conn.close()

    form_counts = Counter(str(r[1]).strip() for r in raw if r[1] is not None and str(r[1]).strip())
    forms = sorted([f for f, n in form_counts.items() if n >= args.min_form_count])
    form_map = {f: i for i, f in enumerate(forms)}
    author_values = sorted({str(r[2]).strip() for r in raw if len(r) > 2 and r[2] is not None and str(r[2]).strip()})
    author_map = {a: i for i, a in enumerate(author_values)}

    rows = []
    for i, r in enumerate(raw):
        text, cipai = r[0], str(r[1]).strip() if r[1] is not None else ""
        if cipai not in form_map or not text:
            continue
        lines = normalize_lines(text)
        chars = chars_from_lines(lines)
        if len(chars) < 4:
            continue
        author = str(r[2]).strip() if len(r) > 2 and r[2] is not None else ""
        tone, rhyme = labels_from_lexicon(chars, lexicon)
        pid = f"ccsc:{i}"
        rows.append({
            "poem_id": pid,
            "lines": lines,
            "form_label": form_map[cipai],
            # Author is optional analysis metadata; manuscript says it is not a prosodic label.
            "style_label": author_map.get(author, -100),
            "tone_labels": tone,
            "rhyme_labels": rhyme,
            "phonology_candidates": candidates_for(chars, lexicon),
            "tone_edges": [],
            "rhyme_positive_pairs": [],
            "rhyme_negative_pairs": [],
            "_split": deterministic_split(pid),
        })

    counts = write_splits(rows, args.out)
    out = Path(args.out)
    (out / "label_maps.json").write_text(json.dumps({"form": form_map, "author": author_map}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CCSC processed counts:", counts)
    print("Detected table:", table, "database:", db)
    print("NOTE: deterministic 80/10/10 splits replace the unavailable exact paper split.")


if __name__ == "__main__":
    main()
