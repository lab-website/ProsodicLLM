#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from prosodicllm.config import load_config
from prosodicllm.data import PoetryJsonlDataset, ProsodicCollator, IGNORE
from prosodicllm.metrics import classification_metrics, tone_error_rate, rhyme_pair_f1, wilson_interval
from prosodicllm.model import ProsodicLLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ablation = ckpt.get("ablation")

    tok = AutoTokenizer.from_pretrained(cfg["model"]["backbone_name"],
                                        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)))
    ds = PoetryJsonlDataset(cfg["data"][f"{args.split}_path"])
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=ProsodicCollator(tok, int(cfg["training"]["max_length"])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProsodicLLM(cfg, ablation=ablation).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    form_t, form_p, style_t, style_p = [], [], [], []
    tone_t, tone_p, rhyme_t, rhyme_p = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dl):
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            out = model(batch)
            form_t.extend(batch["form_labels"].cpu().tolist())
            form_p.extend(out["form_logits"].argmax(-1).cpu().tolist())
            style_t.extend(batch["style_labels"].cpu().tolist())
            style_p.extend(out["style_logits"].argmax(-1).cpu().tolist())
            C = int(batch["char_mask"][0].sum().item())
            gt_t = batch["tone_labels"][0, :C].cpu().tolist()
            pr_t = out["tone_logits"][0, :C].argmax(-1).cpu().tolist()
            gt_r = batch["rhyme_labels"][0, :C].cpu().tolist()
            pr_r = out["rhyme_logits"][0, :C].argmax(-1).cpu().tolist()
            tone_t.extend(gt_t); tone_p.extend(pr_t)
            rhyme_t.append(gt_r); rhyme_p.append(pr_r)

    metrics = {}
    metrics.update(classification_metrics(form_t, form_p, "form_"))
    metrics.update(classification_metrics(style_t, style_p, "style_"))
    metrics["TER"] = tone_error_rate(tone_t, tone_p)
    metrics["R-F1"] = rhyme_pair_f1(rhyme_t, rhyme_p)
    valid_form = [(t, p) for t, p in zip(form_t, form_p) if t != IGNORE]
    if valid_form:
        correct = sum(int(t == p) for t, p in valid_form)
        metrics["form_accuracy_wilson95"] = wilson_interval(correct, len(valid_form))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
