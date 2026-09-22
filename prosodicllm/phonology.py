from __future__ import annotations

import math
import torch
from torch import nn


class PhonologyEncoder(nn.Module):
    """Soft reading selector corresponding to the manuscript's Eq. (1).

    Candidate categorical features are embedded, projected to d_p, and scored against
    an initial character representation. Multiple readings remain active via softmax.
    """

    def __init__(self, hidden_size: int, d_p: int = 64, hidden_width: int = 256,
                 max_rhyme_classes: int = 256):
        super().__init__()
        e = 16
        self.hist_tone_emb = nn.Embedding(8, e)
        self.modern_tone_emb = nn.Embedding(8, e)
        self.rhyme_emb = nn.Embedding(max_rhyme_classes, e)
        self.poly_emb = nn.Embedding(2, e)
        self.feature_mlp = nn.Sequential(
            nn.Linear(e * 4, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, d_p),
        )
        self.to_query_space = nn.Linear(d_p, hidden_size, bias=False)
        self.scale = math.sqrt(hidden_size)

    def forward(self, candidate_feats: torch.Tensor, candidate_mask: torch.Tensor,
                h0_char: torch.Tensor):
        # candidate_feats: [B,C,R,4], h0_char: [B,C,H]
        ht = candidate_feats[..., 0].clamp(0, self.hist_tone_emb.num_embeddings - 1)
        mt = candidate_feats[..., 1].clamp(0, self.modern_tone_emb.num_embeddings - 1)
        rh = candidate_feats[..., 2].clamp(0, self.rhyme_emb.num_embeddings - 1)
        po = candidate_feats[..., 3].clamp(0, 1)
        f = torch.cat([
            self.hist_tone_emb(ht), self.modern_tone_emb(mt),
            self.rhyme_emb(rh), self.poly_emb(po)
        ], dim=-1)
        cand = self.feature_mlp(f)  # [B,C,R,d_p]
        query = self.to_query_space(cand)  # [B,C,R,H]
        scores = (query * h0_char.unsqueeze(2)).sum(-1) / self.scale
        scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)
        alpha = torch.where(candidate_mask, alpha, torch.zeros_like(alpha))
        denom = alpha.sum(-1, keepdim=True).clamp_min(1e-8)
        alpha = alpha / denom
        p = (alpha.unsqueeze(-1) * cand).sum(dim=2)
        return p, alpha
