from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


IGNORE = -100


class PoetryJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def poem_chars(lines: List[str]) -> tuple[List[str], List[int]]:
    chars, line_ids = [], []
    punct = set("，。！？；：、,.!?;:‘’“”（）()《》〈〉—… \t\r\n")
    for li, line in enumerate(lines):
        for ch in line:
            if ch in punct or ch.isspace():
                continue
            chars.append(ch)
            line_ids.append(li)
    return chars, line_ids


class ProsodicCollator:
    """Character-preserving collator.

    Each Chinese character is tokenized separately. If one character becomes multiple
    sub-tokens, token_to_char records the mapping so character hidden states can be
    mean-pooled and phonology vectors can be expanded back to token positions.
    """

    def __init__(self, tokenizer, max_length: int = 512, max_readings: int = 4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_readings = max_readings
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    def _encode_chars(self, chars: List[str]):
        ids: List[int] = []
        mapping: List[int] = []
        if self.tokenizer.bos_token_id is not None:
            ids.append(self.tokenizer.bos_token_id)
            mapping.append(-1)
        for ci, ch in enumerate(chars):
            piece = self.tokenizer.encode(ch, add_special_tokens=False)
            if not piece:
                piece = [self.tokenizer.unk_token_id]
            if len(ids) + len(piece) > self.max_length:
                break
            ids.extend(piece)
            mapping.extend([ci] * len(piece))
        return ids, mapping

    @staticmethod
    def _candidate_tuple(c: Dict[str, Any]):
        return [
            int(c.get("hist_tone", 0)),
            int(c.get("modern_tone", 0)),
            int(c.get("rhyme", 0)),
            int(c.get("polyphonic", 0)),
        ]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded = []
        max_t, max_c = 0, 0
        for row in batch:
            chars, inferred_line_ids = poem_chars(row["lines"])
            ids, mapping = self._encode_chars(chars)
            kept_chars = (max(mapping) + 1) if mapping and max(mapping) >= 0 else 0
            chars = chars[:kept_chars]
            inferred_line_ids = inferred_line_ids[:kept_chars]
            encoded.append((row, chars, inferred_line_ids, ids, mapping))
            max_t = max(max_t, len(ids))
            max_c = max(max_c, len(chars))

        B = len(batch)
        input_ids = torch.full((B, max_t), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_t), dtype=torch.long)
        token_to_char = torch.full((B, max_t), -1, dtype=torch.long)
        char_mask = torch.zeros((B, max_c), dtype=torch.bool)
        candidate_feats = torch.zeros((B, max_c, self.max_readings, 4), dtype=torch.long)
        candidate_mask = torch.zeros((B, max_c, self.max_readings), dtype=torch.bool)
        tone_labels = torch.full((B, max_c), IGNORE, dtype=torch.long)
        rhyme_labels = torch.full((B, max_c), IGNORE, dtype=torch.long)
        form_labels = torch.full((B,), IGNORE, dtype=torch.long)
        style_labels = torch.full((B,), IGNORE, dtype=torch.long)

        line_ids_batch, tone_edges, rhyme_pos, rhyme_neg, poem_ids = [], [], [], [], []

        for b, (row, chars, inferred_line_ids, ids, mapping) in enumerate(encoded):
            T, C = len(ids), len(chars)
            input_ids[b, :T] = torch.tensor(ids)
            attention_mask[b, :T] = 1
            token_to_char[b, :T] = torch.tensor(mapping)
            char_mask[b, :C] = True
            poem_ids.append(row.get("poem_id", str(b)))

            line_ids = row.get("line_ids", inferred_line_ids)[:C]
            line_ids_batch.append(torch.tensor(line_ids, dtype=torch.long))

            cands = row.get("phonology_candidates", [])
            for ci in range(C):
                per_char = cands[ci] if ci < len(cands) and cands[ci] else [
                    {"hist_tone": 0, "modern_tone": 0, "rhyme": 0, "polyphonic": 0}
                ]
                for ri, cand in enumerate(per_char[: self.max_readings]):
                    candidate_feats[b, ci, ri] = torch.tensor(self._candidate_tuple(cand))
                    candidate_mask[b, ci, ri] = True

            tl = row.get("tone_labels", [])[:C]
            rl = row.get("rhyme_labels", [])[:C]
            if tl:
                tone_labels[b, : len(tl)] = torch.tensor(tl, dtype=torch.long)
            if rl:
                rhyme_labels[b, : len(rl)] = torch.tensor(rl, dtype=torch.long)

            form_labels[b] = int(row.get("form_label", IGNORE))
            style_labels[b] = int(row.get("style_label", IGNORE))

            def valid_pairs(name):
                out = []
                for uv in row.get(name, []):
                    if len(uv) >= 2 and uv[0] < C and uv[1] < C:
                        out.append((int(uv[0]), int(uv[1])))
                return out

            tone_edges.append(valid_pairs("tone_edges"))
            rhyme_pos.append(valid_pairs("rhyme_positive_pairs"))
            rhyme_neg.append(valid_pairs("rhyme_negative_pairs"))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_to_char": token_to_char,
            "char_mask": char_mask,
            "candidate_feats": candidate_feats,
            "candidate_mask": candidate_mask,
            "tone_labels": tone_labels,
            "rhyme_labels": rhyme_labels,
            "form_labels": form_labels,
            "style_labels": style_labels,
            "line_ids": line_ids_batch,
            "tone_edges": tone_edges,
            "rhyme_pos_pairs": rhyme_pos,
            "rhyme_neg_pairs": rhyme_neg,
            "poem_ids": poem_ids,
        }
