from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn


REL_CHAR_LINE = 0
REL_LINE_CHAR = 1
REL_LINE_POEM = 2
REL_POEM_LINE = 3
REL_TONE = 4
REL_RHYME = 5
NUM_RELATIONS = 6


class HierarchicalProsodyGraph(nn.Module):
    """Hierarchical character-line-poem graph with learned relation weights.

    The manuscript leaves phi_(u,v) unspecified. Here phi is implemented as a learned
    scalar per relation type, which keeps the message-passing equation explicit and
    makes the assumption easy to replace.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rel_weight = nn.Parameter(torch.ones(NUM_RELATIONS))
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

    @staticmethod
    def _build_edges(line_ids: torch.Tensor,
                     tone_edges: Sequence[Tuple[int, int]],
                     rhyme_pairs: Sequence[Tuple[int, int]]):
        C = int(line_ids.numel())
        n_lines = int(line_ids.max().item()) + 1 if C else 0
        line_offset = C
        poem_node = C + n_lines
        edges: List[Tuple[int, int, int]] = []

        for ci in range(C):
            li = line_offset + int(line_ids[ci].item())
            edges.extend([(ci, li, REL_CHAR_LINE), (li, ci, REL_LINE_CHAR)])
        for li in range(n_lines):
            ln = line_offset + li
            edges.extend([(ln, poem_node, REL_LINE_POEM), (poem_node, ln, REL_POEM_LINE)])
        for u, v in tone_edges:
            if u < C and v < C:
                edges.extend([(u, v, REL_TONE), (v, u, REL_TONE)])
        for u, v in rhyme_pairs:
            if u < C and v < C:
                edges.extend([(u, v, REL_RHYME), (v, u, REL_RHYME)])
        return edges, n_lines, poem_node

    def forward_one(self, char_features: torch.Tensor, line_ids: torch.Tensor,
                    tone_edges: Sequence[Tuple[int, int]],
                    rhyme_pairs: Sequence[Tuple[int, int]]):
        C = char_features.size(0)
        if C == 0:
            raise ValueError("A poem must contain at least one scored character")
        x_char = self.input_proj(char_features)
        edges, n_lines, poem_node = self._build_edges(line_ids, tone_edges, rhyme_pairs)

        line_feats = []
        for li in range(n_lines):
            mask = line_ids == li
            line_feats.append(x_char[mask].mean(0) if mask.any() else x_char.new_zeros(x_char.size(-1)))
        line_feats = torch.stack(line_feats, 0) if line_feats else x_char.new_zeros((0, x_char.size(-1)))
        poem_feat = line_feats.mean(0, keepdim=True) if n_lines else x_char.mean(0, keepdim=True)
        nodes = torch.cat([x_char, line_feats, poem_feat], dim=0)

        if edges:
            src = torch.tensor([e[0] for e in edges], device=nodes.device, dtype=torch.long)
            dst = torch.tensor([e[1] for e in edges], device=nodes.device, dtype=torch.long)
            rel = torch.tensor([e[2] for e in edges], device=nodes.device, dtype=torch.long)
        else:
            src = dst = rel = torch.empty(0, device=nodes.device, dtype=torch.long)

        for mlp, norm in zip(self.layers, self.norms):
            agg = torch.zeros_like(nodes)
            if src.numel():
                msg = nodes[src] * self.rel_weight[rel].unsqueeze(-1)
                agg.index_add_(0, dst, msg)
                deg = torch.zeros(nodes.size(0), device=nodes.device, dtype=nodes.dtype)
                deg.index_add_(0, dst, torch.ones(dst.numel(), device=nodes.device, dtype=nodes.dtype))
                agg = agg / deg.clamp_min(1).unsqueeze(-1)
            nodes = norm(nodes + mlp(agg))

        return {
            "char": nodes[:C],
            "line": nodes[C:C + n_lines],
            "poem": nodes[poem_node],
        }

    def forward(self, char_features: torch.Tensor, char_mask: torch.Tensor,
                line_ids: List[torch.Tensor], tone_edges, rhyme_pos_pairs):
        B, Cmax, _ = char_features.shape
        char_out = char_features.new_zeros((B, Cmax, self.input_proj.out_features))
        poem_out = char_features.new_zeros((B, self.input_proj.out_features))
        line_out = []
        for b in range(B):
            C = int(char_mask[b].sum().item())
            r = self.forward_one(
                char_features[b, :C],
                line_ids[b].to(char_features.device),
                tone_edges[b],
                rhyme_pos_pairs[b],
            )
            char_out[b, :C] = r["char"]
            poem_out[b] = r["poem"]
            line_out.append(r["line"])
        return char_out, poem_out, line_out
