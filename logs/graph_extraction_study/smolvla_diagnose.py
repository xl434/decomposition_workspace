#!/usr/bin/env python3
"""Get full graph break reasons for the real SmolVLA model.
Run: conda activate lerobot && python explain_smolvla.py
"""
import sys
from pathlib import Path

import torch

LEROBOT_SRC = str(Path(__file__).resolve().parent.parent.parent / "data" / "lerobot" / "src")
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
import torch.nn as nn

class SmolVLAWrapper(nn.Module):
    def __init__(self, vla_model):
        super().__init__()
        self.vla = vla_model
    def forward(self, image, img_mask, lang_tokens, lang_masks, state, actions, noise, time):
        return self.vla.forward(
            images=[image], img_masks=[img_mask],
            lang_tokens=lang_tokens, lang_masks=lang_masks,
            state=state, actions=actions, noise=noise, time=time,
        )

config = SmolVLAConfig(
    vlm_model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    load_vlm_weights=False, freeze_vision_encoder=True,
    train_expert_only=True, attention_mode="cross_attn",
    num_vlm_layers=16, expert_width_multiplier=0.75,
    chunk_size=50, max_state_dim=32, max_action_dim=32,
)
inner = VLAFlowMatching(config)
inner.eval()
model = SmolVLAWrapper(inner)
model.eval()

B = 1
vocab = inner.vlm_with_expert.config.text_config.vocab_size
inputs = [
    torch.randn(B, 3, 512, 512),      # image
    torch.ones(B, dtype=torch.bool),    # img_mask
    torch.randint(0, vocab, (B, 16)),   # lang_tokens
    torch.ones(B, 16, dtype=torch.bool),# lang_masks
    torch.randn(B, 32),                # state
    torch.randn(B, 50, 32),            # actions
    torch.randn(B, 50, 32),            # noise
    torch.rand(B) * 0.999 + 0.001,     # time
]

print("Running torch._dynamo.explain()...")
explanation = torch._dynamo.explain(model)(*inputs)

print(f"\nGraphs: {explanation.graph_count}")
n_breaks = getattr(explanation, 'break_count',
                  getattr(explanation, 'graph_break_count', explanation.graph_count - 1))
print(f"Graph breaks: {n_breaks}")

break_reasons = getattr(explanation, 'break_reasons',
                       getattr(explanation, 'graph_break_reasons', []))

print(f"\n{'='*80}")
print(f"ALL {len(break_reasons)} GRAPH BREAK REASONS (full text)")
print(f"{'='*80}")

for i, reason in enumerate(break_reasons):
    print(f"\n--- Break #{i+1} ---")
    print(str(reason))
    print()

# Also show graph sizes
print(f"\n{'='*80}")
print(f"GRAPH SIZES")
print(f"{'='*80}")
for i, gm in enumerate(explanation.graphs):
    n_ops = sum(1 for n in gm.graph.nodes
               if n.op in ("call_function", "call_module", "call_method"))
    print(f"  Graph {i+1}: {n_ops} ops")
print(f"  Total: {sum(sum(1 for n in gm.graph.nodes if n.op in ('call_function','call_module','call_method')) for gm in explanation.graphs)} ops")
