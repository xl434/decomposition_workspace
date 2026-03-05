"""
Step 3 Refactored: action_time_mlp decomposed into fusion-level children.

Children:
  - action_in_proj: nn.Linear(32, 720) - action projection (L0 kernel)
  - sinusoidal_pos_emb: sinusoidal time embedding (L1 fusion)
  - silu_linear_mlp: Linear+SiLU+Linear MLP (L1 fusion)

forward() only does data plumbing: expand, concatenation.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)

import sinusoidal_pos_emb as sinusoidal_pos_emb_mod
import silu_linear_mlp as silu_linear_mlp_mod

EXPERT_HIDDEN_SIZE = 720
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50


class RefactoredModel(nn.Module):
    """Action-Time MLP decomposed into fusion-level children."""

    def __init__(self):
        super().__init__()
        self.action_in_proj = nn.Linear(MAX_ACTION_DIM, EXPERT_HIDDEN_SIZE)
        self.sinusoidal_pos_emb = sinusoidal_pos_emb_mod.Model()
        self.silu_linear_mlp = silu_linear_mlp_mod.Model()

    def forward(self, noisy_actions, timestep):
        # Project actions
        action_emb = self.action_in_proj(noisy_actions)
        dtype = action_emb.dtype

        # Compute time embedding
        time_emb = self.sinusoidal_pos_emb(timestep)
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)

        # Concatenate and run MLP
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.silu_linear_mlp(action_time_emb)
        return action_time_emb


def get_inputs():
    return [torch.randn(1, CHUNK_SIZE, MAX_ACTION_DIM),
            torch.rand(1) * 0.999 + 0.001]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, CHUNK_SIZE, EXPERT_HIDDEN_SIZE)]

def run_tests():
    try:
        model = RefactoredModel()
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected = get_expected_output_shape()
            assert tuple(output.shape) == tuple(expected[0]), \
                f"Shape mismatch: {output.shape} vs {expected[0]}"
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
