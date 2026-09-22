#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocess_utils import (
    candidates_for, chars_from_lines, deterministic_split, even_line_rhyme_pairs,
    labels_from_lexicon, load_lexicon, normalize_lines, write_splits,
)


def iter_json_objects(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield f"{path.stem}:{i}", obj
                except Exception:
                    pass
    for path in sorted(root.rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        seq = obj if isinstance(obj, list) else obj.get("data", []) if isinstance(obj, dict) else []
        if isinstance(seq, list):
            for i, x in enumerate(seq):
                if isinstance(x, dict):
                    yield f"{path.stem}:{i}", x


def extract_lines(obj):
    for key in ["lines", "paragraphs", "content", "text", "poem"]:
        if key in obj and obj[key]:
            return normalize_lines(obj[key])
    return []


def main():
    ap = argparse.ArgumentParser(description="Conservative THU-CCPC/CRRD preprocessor")
    ap.add_argument("--root", default="data/raw/THUNLP-AIPoet-Datasets")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--lexicon", default=None,
                    help="Optional JSON mapping chars to candidates/gold_tone/gold_rhyme from THU-CRRD alignment")
    ap.add_argument("--infer-even-line-rhyme", action="store_true",
                    help="Heuristic; do NOT use for exact reproduction unless justified by your rule table")
    args = ap.parse_args()
    lexicon = load_lexicon(args.lexicon)
    root = Path(args.root)

    rows = []
    seen = set()
    for pid, obj in iter_json_objects(root):
        lines = extract_lines(obj)
        if not lines:
            continue
        text_key = "|".join(lines)
        if text_key in seen:
            continue
        seen.add(text_key)
        chars = chars_from_lines(lines)
        if len(chars) < 4:
            continue
        tone, rhyme = labels_from_lexicon(chars, lexicon)
        form = obj.get("form_label", obj.get("form_id", -100))
        style = obj.get("style_label", obj.get("style_id", -100))
        row = {
            "poem_id": str(obj.get("id", pid)),
            "lines": lines,
            "form_label": int(form) if str(form).lstrip("-").isdigit() else -100,
            "style_label": int(style) if str(style).lstrip("-").isdigit() else -100,
            "tone_labels": obj.get("tone_labels", tone),
            "rhyme_labels": obj.get("rhyme_labels", rhyme),
            "phonology_candidates": candidates_for(chars, lexicon),
            "tone_edges": obj.get("tone_edges", []),
            "rhyme_positive_pairs": obj.get("rhyme_positive_pairs",
                even_line_rhyme_pairs(lines) if args.infer_even_line_rhyme else []),
            "rhyme_negative_pairs": obj.get("rhyme_negative_pairs", []),
            "_split": deterministic_split(str(obj.get("id", pid))),
        }
        rows.append(row)

    if not rows:
        raise SystemExit(
            "No structured poems were auto-detected. The THU repository layout may differ. "
            "Convert its poem text to JSON/JSONL with fields such as lines/paragraphs/content, "
            "then rerun this script. Exact CRRD alignment is intentionally not guessed."
        )
    counts = write_splits(rows, args.out)
    print("CPP-like processed counts:", counts)
    print("NOTE: exact paper partitions/rule tables were not published in the supplied manuscript.")


if __name__ == "__main__":
    main()
