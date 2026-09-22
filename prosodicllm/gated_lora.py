from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PhonologyState:
    token_features: torch.Tensor | None = None


class GatedLoRALinear(nn.Module):
    """LoRA with token-conditioned rank gates.

    For each token x and phonology vector p:
      delta(x,p) = alpha/r * B( sigmoid(G p) * A(x) )
    """

    def __init__(self, base: nn.Module, state: PhonologyState, d_p: int, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.0, enabled_gate: bool = True):
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError("GatedLoRALinear requires a linear-like module with in_features/out_features")
        self.base = base
        self.state = state
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.enabled_gate = enabled_gate
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.G = nn.Parameter(torch.empty(rank, d_p))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.xavier_uniform_(self.G)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        low = F.linear(self.dropout(x), self.A)
        p = self.state.token_features
        if self.enabled_gate and p is not None:
            gate = torch.sigmoid(F.linear(p.to(dtype=low.dtype), self.G))
            if gate.shape[:-1] != low.shape[:-1]:
                # Generation/cache paths are not used by training; align the current suffix if needed.
                gate = gate[..., -low.shape[-2]:, :]
            low = low * gate
        delta = F.linear(low, self.B) * self.scaling
        return y + delta.to(dtype=y.dtype)


class GatedLoRAFusedQKV(nn.Module):
    """Support legacy Qwen fused c_attn by adapting Q and V thirds only."""

    def __init__(self, base: nn.Module, state: PhonologyState, d_p: int, rank: int = 16,
                 alpha: float = 32.0, dropout: float = 0.0, enabled_gate: bool = True):
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError("Fused QKV output dimension must be divisible by 3")
        self.base = base
        self.state = state
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.enabled_gate = enabled_gate
        third = base.out_features // 3
        self.third = third
        self.Aq = nn.Parameter(torch.empty(rank, base.in_features))
        self.Bq = nn.Parameter(torch.zeros(third, rank))
        self.Av = nn.Parameter(torch.empty(rank, base.in_features))
        self.Bv = nn.Parameter(torch.zeros(third, rank))
        self.Gq = nn.Parameter(torch.empty(rank, d_p))
        self.Gv = nn.Parameter(torch.empty(rank, d_p))
        for p in (self.Aq, self.Av):
            nn.init.kaiming_uniform_(p, a=5 ** 0.5)
        nn.init.xavier_uniform_(self.Gq)
        nn.init.xavier_uniform_(self.Gv)
        for p in self.base.parameters():
            p.requires_grad = False

    def _branch(self, x, A, B, G):
        low = F.linear(self.dropout(x), A)
        p = self.state.token_features
        if self.enabled_gate and p is not None:
            gate = torch.sigmoid(F.linear(p.to(dtype=low.dtype), G))
            if gate.shape[:-1] != low.shape[:-1]:
                gate = gate[..., -low.shape[-2]:, :]
            low = low * gate
        return F.linear(low, B) * self.scaling

    def forward(self, x):
        y = self.base(x)
        dq = self._branch(x, self.Aq, self.Bq, self.Gq)
        dv = self._branch(x, self.Av, self.Bv, self.Gv)
        z = torch.zeros_like(dq)
        delta = torch.cat([dq, z, dv], dim=-1)
        return y + delta.to(dtype=y.dtype)


def _set_module(root: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parent = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def inject_gated_lora(model: nn.Module, state: PhonologyState, d_p: int, rank: int,
                      alpha: float, dropout: float, targets: Iterable[str],
                      enabled_gate: bool = True) -> List[str]:
    names: List[Tuple[str, nn.Module]] = list(model.named_modules())
    injected = []
    target_set = set(targets)

    # Prefer explicit q_proj/v_proj if present. Only use fused c_attn when those are absent.
    explicit = [n for n, m in names if n.split(".")[-1] in {"q_proj", "v_proj"}
                and hasattr(m, "in_features") and hasattr(m, "out_features")]
    use_fused = not explicit and "c_attn" in target_set

    for name, module in names:
        leaf = name.split(".")[-1]
        if explicit and leaf in {"q_proj", "v_proj"} and leaf in target_set:
            wrapped = GatedLoRALinear(module, state, d_p, rank, alpha, dropout, enabled_gate)
            _set_module(model, name, wrapped)
            injected.append(name)
        elif use_fused and leaf == "c_attn" and hasattr(module, "in_features") and hasattr(module, "out_features"):
            wrapped = GatedLoRAFusedQKV(module, state, d_p, rank, alpha, dropout, enabled_gate)
            _set_module(model, name, wrapped)
            injected.append(name)

    if not injected:
        raise RuntimeError(
            "No compatible Q/V attention projections found. Inspect model.named_modules() and "
            "adjust model.lora_targets or add a checkpoint-specific wrapper."
        )
    return injected
