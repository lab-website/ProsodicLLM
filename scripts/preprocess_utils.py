from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:‘’“”（）()《》〈〉—…\s]+")


def normalize_lines(text_or_lines) -> List[str]:
    if isinstance(text_or_lines, list):
        lines = [str(x).strip() for x in text_or_lines]
    else:
        text = str(text_or_lines).strip()
        lines = [x.strip() for x in re.split(r"[。！？!?\n]+", text) if x.strip()]
    return [x for x in lines if x]


def chars_from_lines(lines: List[str]) -> List[str]:
    return [ch for line in lines for ch in PUNCT_RE.sub("", line)]


def deterministic_split(key: str, train=0.8, val=0.1) -> str:
    v = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if v < train:
        return "train"
    if v < train + val:
        return "val"
    return "test"


def load_lexicon(path: str | None) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def candidates_for(chars: List[str], lexicon: Dict):
    out = []
    for ch in chars:
        entry = lexicon.get(ch, {})
        cands = entry.get("candidates", entry if isinstance(entry, list) else [])
        if not cands:
            cands = [{"hist_tone": 0, "modern_tone": 0, "rhyme": 0, "polyphonic": 0}]
        norm = []
        for c in cands:
            norm.append({
                "hist_tone": int(c.get("hist_tone", 0)),
                "modern_tone": int(c.get("modern_tone", 0)),
                "rhyme": int(c.get("rhyme", 0)),
                "polyphonic": int(c.get("polyphonic", len(cands) > 1)),
            })
        out.append(norm)
    return out


def labels_from_lexicon(chars: List[str], lexicon: Dict):
    tone, rhyme = [], []
    for ch in chars:
        entry = lexicon.get(ch, {})
        if isinstance(entry, dict):
            tone.append(int(entry.get("gold_tone", -100)))
            rhyme.append(int(entry.get("gold_rhyme", -100)))
        else:
            tone.append(-100); rhyme.append(-100)
    return tone, rhyme


def even_line_rhyme_pairs(lines: List[str]):
    """Optional heuristic only; disabled by default in preprocessing scripts."""
    offsets, cur = [], 0
    for line in lines:
        clean = PUNCT_RE.sub("", line)
        cur += len(clean)
        offsets.append(cur - 1)
    ends = [offsets[i] for i in range(1, len(offsets), 2)]
    return [[ends[i], ends[j]] for i in range(len(ends)) for j in range(i + 1, len(ends))]


def write_splits(rows: Iterable[dict], out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    handles = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ["train", "val", "test"]}
    counts = {s: 0 for s in handles}
    try:
        for row in rows:
            split = row.pop("_split", deterministic_split(row["poem_id"]))
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for f in handles.values():
            f.close()
    return counts
