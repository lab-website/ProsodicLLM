#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from prosodicllm.config import load_config
from prosodicllm.data import PoetryJsonlDataset, ProsodicCollator
from prosodicllm.losses import compose_total_loss
from prosodicllm.model import ProsodicLLM
from prosodicllm.utils import ensure_dir, set_seed


def cosine_with_warmup(optimizer, total_steps, warmup_ratio):
    warm = max(1, int(total_steps * warmup_ratio))
    def fn(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ablation", default=None,
                    choices=[None, "no_graph", "no_contrastive", "standard_lora", "no_phonology"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    tr = cfg["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(tr["gradient_accumulation_steps"]),
        mixed_precision=str(tr.get("mixed_precision", "bf16")),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["backbone_name"],
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    collator = ProsodicCollator(tokenizer, max_length=int(tr["max_length"]))
    train_ds = PoetryJsonlDataset(cfg["data"]["train_path"])
    val_ds = PoetryJsonlDataset(cfg["data"]["val_path"])
    train_dl = DataLoader(train_ds, batch_size=int(tr["per_device_batch_size"]), shuffle=True,
                          num_workers=int(tr["num_workers"]), collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=int(tr["per_device_batch_size"]), shuffle=False,
                        num_workers=int(tr["num_workers"]), collate_fn=collator)

    model = ProsodicLLM(cfg, ablation=args.ablation)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=float(tr["learning_rate"]), weight_decay=float(tr["weight_decay"]))
    updates_per_epoch = math.ceil(len(train_dl) / int(tr["gradient_accumulation_steps"]))
    total_steps = max(1, updates_per_epoch * int(tr["epochs"]))
    scheduler = cosine_with_warmup(optimizer, total_steps, float(tr["warmup_ratio"]))

    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )
    outdir = ensure_dir(tr["output_dir"] + (f"_{args.ablation}" if args.ablation else ""))
    best = float("inf")

    if accelerator.is_main_process:
        accelerator.print("trainable params:", accelerator.unwrap_model(model).trainable_parameter_summary())
        accelerator.print("LoRA modules:", accelerator.unwrap_model(model).injected_lora[:8], "...")

    for epoch in range(1, int(tr["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_dl, disable=not accelerator.is_local_main_process, desc=f"train {epoch}")
        for step, batch in enumerate(pbar, 1):
            with accelerator.accumulate(model):
                out = model(batch)
                loss, parts = compose_total_loss(out, batch, cfg["loss"])
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % int(tr["log_every"]) == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, disable=not accelerator.is_local_main_process, desc=f"val {epoch}"):
                out = model(batch)
                loss, _ = compose_total_loss(out, batch, cfg["loss"])
                gathered = accelerator.gather_for_metrics(loss.detach().reshape(1))
                val_sum += gathered.sum().item()
                val_n += gathered.numel()
        val_loss = val_sum / max(1, val_n)
        accelerator.print(f"epoch={epoch} val_loss={val_loss:.6f}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            raw = accelerator.unwrap_model(model)
            state = {k: v.cpu() for k, v in raw.state_dict().items() if v.requires_grad or
                     any(tag in k for tag in ["A", "B", "G", "phonology_encoder", "graph", "projection", "head", "phonology_to_hidden", "no_graph_proj"])}
            torch.save({"epoch": epoch, "model": state, "config": cfg, "ablation": args.ablation}, outdir / "last.pt")
            if val_loss < best:
                best = val_loss
                torch.save({"epoch": epoch, "model": state, "config": cfg, "ablation": args.ablation}, outdir / "best.pt")


if __name__ == "__main__":
    main()
