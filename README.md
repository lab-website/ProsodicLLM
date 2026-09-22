# ProsodicLLM reproduction (multi-file PyTorch implementation)

This repository reconstructs the architecture described in the supplied manuscript:
**ProsodicLLM: Constraint-Aware LLM for Classical Chinese Poetry Prosody and Style Analysis**.

## What is implemented

- Qwen backbone with frozen base weights.
- Token-conditioned phonology-grounded gated LoRA.
- Soft candidate-reading selection for polyphonic characters.
- Hierarchical character -> line -> poem graph construction.
- Relation-aware graph message passing.
- Soft tone-consistency and rhyme-pair regularizers.
- Separate prosody/form and style projections.
- Supervised contrastive loss.
- Multi-task heads for tone, rhyme, form and style.
- Accuracy, macro-F1, tone error rate (TER), and rhyme-pair F1 (R-F1).
- Ablation switches for graph, contrastive loss, phonological encoder, and gated LoRA.
- Dataset preprocessing hooks for THU-CCPC/THU-CRRD and CCSC.

## Important reproducibility boundary

The manuscript explicitly states that the exact processed corpora, some validation traces, per-example predictions, independent multi-seed checkpoints, and a pure Qwen-7B + standard-LoRA baseline were not retained. Therefore, this repository reproduces the **reported method and training recipe**, but it cannot guarantee the exact reported numbers without the original processed splits and rule tables.

The paper gives the following retained training settings, which are used in `configs/base.yaml`: LoRA rank 16, LoRA alpha 32, phonology dimension 64, 3 graph layers, graph width 512, contrastive temperature 0.07, loss weights 0.5/0.5/1.0, AdamW lr=3e-4, weight decay=1e-2, cosine decay, 5% warmup, 30 epochs, and global batch size 256.

## Project layout

```text
prosodicllm_repro/
├── configs/base.yaml
├── prosodicllm/
│   ├── config.py
│   ├── data.py
│   ├── phonology.py
│   ├── gated_lora.py
│   ├── graph.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── scripts/
│   ├── download_data.py
│   ├── preprocess_cpp.py
│   ├── preprocess_ccsc.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_ablation.py
├── demo/create_toy_data.py
└── tests/test_smoke.py
```

## Expected processed JSONL schema

Each line is one poem:

```json
{
  "poem_id": "p1",
  "lines": ["春眠不觉晓", "处处闻啼鸟"],
  "form_label": 0,
  "style_label": 0,
  "tone_labels": [0,1,1,0,1, 1,1,0,1,0],
  "rhyme_labels": [-100,-100,-100,-100,3, -100,-100,-100,-100,3],
  "phonology_candidates": [
    [{"hist_tone":0,"modern_tone":1,"rhyme":3,"polyphonic":0}],
    [{"hist_tone":1,"modern_tone":2,"rhyme":5,"polyphonic":1},
     {"hist_tone":0,"modern_tone":4,"rhyme":7,"polyphonic":1}]
  ],
  "tone_edges": [[0,5],[1,6]],
  "rhyme_positive_pairs": [[4,9]],
  "rhyme_negative_pairs": []
}
```

`-100` means "ignore for supervised loss". Character indices exclude punctuation and follow concatenated line order.

## Data

The manuscript points to:

- THU Chinese Classical Poetry / rhythm-rhyme resources: `https://github.com/THUNLP-AIPoet/Datasets`
- Song Ci Corpus (CCSC): `https://doi.org/10.5281/zenodo.17798065`

Run:

```bash
python scripts/download_data.py --root data/raw
```

The exact original preprocessing/rule table is not available in the manuscript, so `preprocess_cpp.py` and `preprocess_ccsc.py` are intentionally conservative. They convert discoverable text/SQLite metadata to the common JSONL format and leave unknown prosodic fields as `-100`. If you have the authors' processed labels or THU-CRRD mapping tables, pass them through `--lexicon` / `--rule-table` to recover the intended graph supervision.

## Quick smoke test

```bash
pip install -r requirements.txt
pip install -e .
python demo/create_toy_data.py
python -m pytest -q
```

## Training

Single process:

```bash
python scripts/train.py --config configs/base.yaml
```

Multi-GPU with Accelerate:

```bash
accelerate config
accelerate launch scripts/train.py --config configs/base.yaml
```

The manuscript reports four NVIDIA A100 80GB GPUs. The per-device micro-batch and gradient accumulation values are implementation details because they were not specified; set them so the effective global batch is 256:

```text
global_batch = per_device_batch * world_size * grad_accumulation
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/base.yaml \
  --checkpoint outputs/prosodicllm/best.pt \
  --split test
```

## Ablations

```bash
python scripts/run_ablation.py --config configs/base.yaml --name no_graph
python scripts/run_ablation.py --config configs/base.yaml --name no_contrastive
python scripts/run_ablation.py --config configs/base.yaml --name standard_lora
python scripts/run_ablation.py --config configs/base.yaml --name no_phonology
```

`standard_lora` disables the token-conditioned gate but keeps graph and contrastive modules, matching the scope described in the manuscript.

## Notes on Qwen compatibility

The paper names Qwen-7B and says LoRA is applied to query/value projections. Modern Qwen-family checkpoints expose `q_proj`/`v_proj`; the original Qwen implementation may expose a fused `c_attn`. This code supports both patterns. For fused QKV, the gated update is injected into the Q and V thirds while K is left unchanged.

If access to the exact legacy checkpoint becomes difficult, you can switch `backbone_name` to a compatible Qwen checkpoint for engineering validation, but that is **not** an exact paper reproduction.
