from __future__ import annotations

import torch
from torch import nn

from prosodicllm.gated_lora import GatedLoRALinear, PhonologyState
from prosodicllm.graph import HierarchicalProsodyGraph
from prosodicllm.losses import supervised_contrastive_loss, tone_relation_loss, rhyme_pair_loss


def test_gated_lora_shape_and_grad():
    base = nn.Linear(8, 8, bias=False)
    state = PhonologyState(torch.randn(2, 5, 4))
    layer = GatedLoRALinear(base, state, d_p=4, rank=2, alpha=4)
    x = torch.randn(2, 5, 8)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert layer.A.grad is not None
    assert layer.B.grad is not None
    assert layer.G.grad is not None
    assert base.weight.grad is None


def test_graph_shapes():
    graph = HierarchicalProsodyGraph(input_dim=16, hidden_dim=12, layers=3)
    chars = torch.randn(2, 8, 16)
    mask = torch.tensor([[1]*8, [1]*6 + [0]*2], dtype=torch.bool)
    line_ids = [torch.tensor([0,0,0,0,1,1,1,1]), torch.tensor([0,0,0,1,1,1])]
    tone_edges = [[(0,4)], [(0,3)]]
    rhyme = [[(3,7)], [(2,5)]]
    c, p, lines = graph(chars, mask, line_ids, tone_edges, rhyme)
    assert c.shape == (2, 8, 12)
    assert p.shape == (2, 12)
    assert len(lines) == 2


def test_losses_finite():
    z = torch.randn(4, 8)
    labels = torch.tensor([0,0,1,1])
    assert torch.isfinite(supervised_contrastive_loss(z, labels))
    tone_logits = torch.randn(2, 5, 2)
    assert torch.isfinite(tone_relation_loss(tone_logits, [[(0,1)], [(2,3)]], 0.1))
    rhyme_vec = torch.randn(2, 5, 8)
    assert torch.isfinite(rhyme_pair_loss(rhyme_vec, [[(0,1)], []], [[], [(2,3)]]))
