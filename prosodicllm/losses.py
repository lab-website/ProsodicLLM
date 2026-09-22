from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

IGNORE = -100


def safe_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid])


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.07) -> torch.Tensor:
    valid = labels != IGNORE
    z, labels = z[valid], labels[valid]
    if z.size(0) < 2:
        return z.sum() * 0.0
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    eye = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = same.sum(1)
    valid_anchor = positives > 0
    if not valid_anchor.any():
        return z.sum() * 0.0
    per_anchor = -(log_prob.masked_fill(~same, 0.0).sum(1) / positives.clamp_min(1))
    return per_anchor[valid_anchor].mean()


def tone_relation_loss(tone_logits: torch.Tensor,
                       edges: List[Sequence[Tuple[int, int]]],
                       margin: float = 0.10) -> torch.Tensor:
    """Soft parity loss from the manuscript.

    s is a continuous ping/ze score in [-1,1]. Each listed edge is interpreted as an
    opposite-tone relation, matching max(0, s_u s_v + epsilon).
    """
    if tone_logits.size(-1) != 2:
        raise ValueError("tone_relation_loss expects 2 tone classes (ping/ze)")
    prob = torch.softmax(tone_logits, dim=-1)
    s = prob[..., 1] - prob[..., 0]
    vals = []
    for b, pairs in enumerate(edges):
        for u, v in pairs:
            if u < s.size(1) and v < s.size(1):
                vals.append(torch.relu(s[b, u] * s[b, v] + margin))
    return torch.stack(vals).mean() if vals else tone_logits.sum() * 0.0


def rhyme_pair_loss(rhyme_vectors: torch.Tensor,
                    pos_pairs: List[Sequence[Tuple[int, int]]],
                    neg_pairs: List[Sequence[Tuple[int, int]]]) -> torch.Tensor:
    vals = []
    z = F.normalize(rhyme_vectors, dim=-1)
    for b, pairs in enumerate(pos_pairs):
        for u, v in pairs:
            if u < z.size(1) and v < z.size(1):
                vals.append(1.0 - (z[b, u] * z[b, v]).sum())
    for b, pairs in enumerate(neg_pairs):
        for u, v in pairs:
            if u < z.size(1) and v < z.size(1):
                vals.append(torch.relu((z[b, u] * z[b, v]).sum()))
    return torch.stack(vals).mean() if vals else rhyme_vectors.sum() * 0.0


def compose_total_loss(outputs, batch, cfg):
    ce_tone = safe_cross_entropy(outputs["tone_logits"], batch["tone_labels"])
    ce_rhyme = safe_cross_entropy(outputs["rhyme_logits"], batch["rhyme_labels"])
    ce_form = safe_cross_entropy(outputs["form_logits"], batch["form_labels"])
    ce_style = safe_cross_entropy(outputs["style_logits"], batch["style_labels"])
    ce = ce_tone + ce_rhyme + cfg["lambda_form"] * ce_form + cfg["lambda_style"] * ce_style

    l_tone = tone_relation_loss(outputs["tone_logits"], batch["tone_edges"], cfg["tone_margin"])
    l_rhyme = rhyme_pair_loss(outputs["rhyme_vectors"], batch["rhyme_pos_pairs"], batch["rhyme_neg_pairs"])
    l_scon = supervised_contrastive_loss(
        outputs["prosody_projection"], batch["form_labels"], outputs["temperature"]
    ) if outputs.get("use_contrastive", True) else ce * 0.0

    total = ce + cfg["lambda_tone"] * l_tone + cfg["lambda_rhyme"] * l_rhyme + cfg["lambda_scon"] * l_scon
    return total, {
        "ce": ce.detach(),
        "tone_reg": l_tone.detach(),
        "rhyme_reg": l_rhyme.detach(),
        "scon": l_scon.detach(),
        "total": total.detach(),
    }
