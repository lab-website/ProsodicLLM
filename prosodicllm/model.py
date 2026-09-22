from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from transformers import AutoModel

from .phonology import PhonologyEncoder
from .gated_lora import PhonologyState, inject_gated_lora
from .graph import HierarchicalProsodyGraph


def _torch_dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(str(name).lower(), torch.bfloat16)


def pool_tokens_to_chars(x: torch.Tensor, token_to_char: torch.Tensor, cmax: int) -> torch.Tensor:
    B, T, H = x.shape
    out = x.new_zeros((B, cmax, H))
    counts = x.new_zeros((B, cmax, 1))
    for b in range(B):
        idx = token_to_char[b]
        valid = idx >= 0
        if valid.any():
            ids = idx[valid].clamp_max(cmax - 1)
            out[b].index_add_(0, ids, x[b, valid])
            counts[b].index_add_(0, ids, torch.ones((ids.numel(), 1), device=x.device, dtype=x.dtype))
    return out / counts.clamp_min(1)


def expand_chars_to_tokens(p_char: torch.Tensor, token_to_char: torch.Tensor) -> torch.Tensor:
    B, T = token_to_char.shape
    D = p_char.size(-1)
    out = p_char.new_zeros((B, T, D))
    for b in range(B):
        valid = token_to_char[b] >= 0
        if valid.any():
            out[b, valid] = p_char[b, token_to_char[b, valid]]
    return out


class ProsodicLLM(nn.Module):
    def __init__(self, cfg: Dict, ablation: str | None = None):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        self.ablation = ablation
        self.use_graph = ablation != "no_graph"
        self.use_contrastive = ablation != "no_contrastive"
        self.use_phonology = ablation != "no_phonology"
        self.gated_lora = ablation != "standard_lora"

        self.backbone = AutoModel.from_pretrained(
            m["backbone_name"],
            trust_remote_code=bool(m.get("trust_remote_code", False)),
            torch_dtype=_torch_dtype(m.get("torch_dtype", "bfloat16")),
        )
        if m.get("freeze_backbone", True):
            for p in self.backbone.parameters():
                p.requires_grad = False
        if m.get("gradient_checkpointing", True) and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.backbone.config, "n_embd", None)
        if hidden is None:
            raise AttributeError("Backbone config exposes neither hidden_size nor n_embd")
        hidden = int(hidden)
        dp = int(m["phonology_dim"])
        self.phonology_state = PhonologyState()
        self.phonology_encoder = PhonologyEncoder(
            hidden_size=hidden,
            d_p=dp,
            hidden_width=int(m["phonology_hidden"]),
            max_rhyme_classes=max(256, int(m["num_rhyme_classes"]) + 1),
        )
        self.injected_lora = inject_gated_lora(
            self.backbone,
            self.phonology_state,
            d_p=dp,
            rank=int(m["lora_rank"]),
            alpha=float(m["lora_alpha"]),
            dropout=float(m.get("lora_dropout", 0.0)),
            targets=m.get("lora_targets", ["q_proj", "v_proj"]),
            enabled_gate=self.gated_lora,
        )

        self.phonology_to_hidden = nn.Linear(dp, hidden, bias=False)
        graph_in = hidden
        self.graph = HierarchicalProsodyGraph(
            input_dim=graph_in,
            hidden_dim=int(m["graph_hidden"]),
            layers=int(m["graph_layers"]),
        )
        self.no_graph_proj = nn.Sequential(
            nn.Linear(hidden, int(m["graph_hidden"])),
            nn.LayerNorm(int(m["graph_hidden"])),
        )
        gh = int(m["graph_hidden"])
        pd = int(m["proj_dim"])
        self.prosody_projection = nn.Sequential(nn.Linear(gh, pd), nn.GELU(), nn.Linear(pd, pd))
        self.style_projection = nn.Sequential(nn.Linear(gh, pd), nn.GELU(), nn.Linear(pd, pd))
        self.tone_head = nn.Linear(gh, int(m["num_tone_classes"]))
        self.rhyme_head = nn.Linear(gh, int(m["num_rhyme_classes"]))
        self.rhyme_vector = nn.Linear(gh, pd)
        self.form_head = nn.Linear(pd, int(m["num_form_classes"]))
        self.style_head = nn.Linear(pd, int(m["num_style_classes"]))
        self.temperature = float(m["contrastive_temperature"])

        # Match custom modules to backbone parameter dtype to avoid bf16/fp32 matmul mismatch.
        dtype = next(self.backbone.parameters()).dtype
        for module in [
            self.phonology_encoder, self.phonology_to_hidden, self.graph, self.no_graph_proj,
            self.prosody_projection, self.style_projection, self.tone_head, self.rhyme_head,
            self.rhyme_vector, self.form_head, self.style_head,
        ]:
            module.to(dtype=dtype)

    def trainable_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "ratio": trainable / max(total, 1)}

    def forward(self, batch: Dict):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_to_char = batch["token_to_char"]
        cmax = batch["char_mask"].size(1)

        # h^(0): frozen token embedding pooled to characters, used to score candidate readings.
        token_h0 = self.backbone.get_input_embeddings()(input_ids)
        char_h0 = pool_tokens_to_chars(token_h0, token_to_char, cmax)
        if self.use_phonology:
            p_char, reading_weights = self.phonology_encoder(
                batch["candidate_feats"], batch["candidate_mask"], char_h0
            )
        else:
            p_char = char_h0.new_zeros((*char_h0.shape[:2], self.cfg["model"]["phonology_dim"]))
            reading_weights = batch["candidate_mask"].to(char_h0.dtype)
            reading_weights = reading_weights / reading_weights.sum(-1, keepdim=True).clamp_min(1)

        p_token = expand_chars_to_tokens(p_char, token_to_char)
        self.phonology_state.token_features = p_token
        try:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        finally:
            self.phonology_state.token_features = None

        token_hidden = out.last_hidden_state
        char_hidden = pool_tokens_to_chars(token_hidden, token_to_char, cmax)
        char_input = char_hidden + self.phonology_to_hidden(p_char).to(char_hidden.dtype)

        if self.use_graph:
            char_graph, poem_graph, line_graph = self.graph(
                char_input, batch["char_mask"], batch["line_ids"],
                batch["tone_edges"], batch["rhyme_pos_pairs"]
            )
        else:
            char_graph = self.no_graph_proj(char_input)
            mask = batch["char_mask"].unsqueeze(-1).to(char_graph.dtype)
            poem_graph = (char_graph * mask).sum(1) / mask.sum(1).clamp_min(1)
            line_graph = []

        z_pros = self.prosody_projection(poem_graph)
        z_style = self.style_projection(poem_graph)
        return {
            "tone_logits": self.tone_head(char_graph),
            "rhyme_logits": self.rhyme_head(char_graph),
            "rhyme_vectors": self.rhyme_vector(char_graph),
            "form_logits": self.form_head(z_pros),
            "style_logits": self.style_head(z_style),
            "prosody_projection": z_pros,
            "style_projection": z_style,
            "reading_weights": reading_weights,
            "line_graph": line_graph,
            "temperature": self.temperature,
            "use_contrastive": self.use_contrastive,
        }
