"""
Component: Action-Time MLP Embedding
Abstraction Level: layer
Parent: VLAFlowMatching (root)

Operations: Linear projection, sinusoidal positional encoding, concatenation, Linear, SiLU, Linear

Input Shapes:
  - noisy_actions: [B, 50, 32] float32
  - timestep: [B] float32

Output Shapes:
  - action_time_emb: [B, 50, 720] float32

Weight Shapes:
  - action_in_proj: Linear(32, 720)
  - action_time_mlp_in: Linear(1440, 720)
  - action_time_mlp_out: Linear(720, 720)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_HIDDEN_SIZE = 720
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = torch.float64 if device != "mps" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


class Model(nn.Module):
    """Action-Time MLP: projects noisy actions and timestep into expert hidden space."""
    def __init__(self):
        super().__init__()
        self.action_in_proj = nn.Linear(MAX_ACTION_DIM, EXPERT_HIDDEN_SIZE)
        self.action_time_mlp_in = nn.Linear(EXPERT_HIDDEN_SIZE * 2, EXPERT_HIDDEN_SIZE)
        self.action_time_mlp_out = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_HIDDEN_SIZE)

    def forward(self, noisy_actions, timestep):
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        dtype = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep, EXPERT_HIDDEN_SIZE, MIN_PERIOD, MAX_PERIOD, device=device
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
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
        model = Model()
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected = get_expected_output_shape()
            assert tuple(output.shape) == tuple(expected[0]), f"Shape mismatch: {output.shape} vs {expected[0]}"
            print(f"Input shapes: {[x.shape for x in inputs]}")
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
