#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def make_row(i: int):
    lines = ["春眠不觉晓", "处处闻啼鸟", "夜来风雨声", "花落知多少"]
    n = 20
    # Toy labels are synthetic and only for pipeline debugging.
    tone = [(j + i) % 2 for j in range(n)]
    rhyme = [-100] * n
    for j in [9, 19]:
        rhyme[j] = 3
    candidates = []
    for j in range(n):
        base = {"hist_tone": tone[j], "modern_tone": (j % 4) + 1, "rhyme": 3 if j in [9, 19] else (j % 8), "polyphonic": 0}
        if j == 2:
            candidates.append([base, {"hist_tone": 1 - tone[j], "modern_tone": 4, "rhyme": 6, "polyphonic": 1}])
        else:
            candidates.append([base])
    return {
        "poem_id": f"toy-{i}",
        "lines": lines,
        "form_label": i % 2,
        "style_label": i % 3,
        "tone_labels": tone,
        "rhyme_labels": rhyme,
        "phonology_candidates": candidates,
        "tone_edges": [[j, j + 5] for j in range(5)] + [[j + 10, j + 15] for j in range(5)],
        "rhyme_positive_pairs": [[9, 19]],
        "rhyme_negative_pairs": [[4, 9]],
    }


def main():
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    for split, n in [("train", 16), ("val", 4), ("test", 4)]:
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i in range(n):
                f.write(json.dumps(make_row(i), ensure_ascii=False) + "\n")
    print("Toy data written to", out.resolve())


if __name__ == "__main__":
    main()
