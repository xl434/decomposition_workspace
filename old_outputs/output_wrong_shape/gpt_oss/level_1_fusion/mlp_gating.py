"""
Component: mlp_gating
Level: 1 (Fusion)
Parent: mlp_block (Level 2)
Children: rms_norm (L0), linear_proj (L0), topk_gate (L0)
Operations: RMSNorm + Gate linear projection + TopK expert selection + Softmax

Input Shapes:
    x: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Output Shapes:
    normed_x: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16
    expert_weights: [SEQ_LEN=16, EXPERTS_PER_TOKEN=2] bfloat16
    expert_indices: [SEQ_LEN=16, EXPERTS_PER_TOKEN=2] int64

Test Configuration:
    HIDDEN_SIZE=128, NUM_EXPERTS=4, EXPERTS_PER_TOKEN=2
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, hidden_size=128, num_experts=4, experts_per_token=2):
        super().__init__()
        self.experts_per_token = experts_per_token
        # RMSNorm
        self.scale = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = 1e-5
        # Gate
        self.gate = nn.Linear(hidden_size, num_experts, dtype=torch.bfloat16)

    def forward(self, x):
        # RMSNorm
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        t = (t * self.scale).to(dtype)
        # Gate
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = F.softmax(experts.values, dim=1)
        return t, expert_weights, experts.indices


def get_inputs():
    return [torch.randn(16, 128, dtype=torch.bfloat16)]


def get_init_inputs():
    return [128, 4, 2]


def get_expected_output_shape():
    return [(16, 128), (16, 2), (16, 2)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        inputs = get_inputs()
        outputs = model(*inputs)

        # Verify output is a tuple of 3 elements
        assert isinstance(outputs, tuple), f"Expected tuple output, got {type(outputs)}"
        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

        expected_shapes = get_expected_output_shape()
        normed_x, expert_weights, expert_indices = outputs

        # Check shapes
        assert normed_x.shape == torch.Size(expected_shapes[0]), \
            f"normed_x shape mismatch: expected {expected_shapes[0]}, got {tuple(normed_x.shape)}"
        assert expert_weights.shape == torch.Size(expected_shapes[1]), \
            f"expert_weights shape mismatch: expected {expected_shapes[1]}, got {tuple(expert_weights.shape)}"
        assert expert_indices.shape == torch.Size(expected_shapes[2]), \
            f"expert_indices shape mismatch: expected {expected_shapes[2]}, got {tuple(expert_indices.shape)}"

        # Check dtypes
        assert normed_x.dtype == torch.bfloat16, \
            f"normed_x dtype mismatch: expected bfloat16, got {normed_x.dtype}"
        assert expert_weights.dtype == torch.bfloat16, \
            f"expert_weights dtype mismatch: expected bfloat16, got {expert_weights.dtype}"
        assert expert_indices.dtype == torch.int64, \
            f"expert_indices dtype mismatch: expected int64, got {expert_indices.dtype}"

        # Check no NaN/Inf in float outputs
        for name, tensor in [("normed_x", normed_x), ("expert_weights", expert_weights)]:
            assert not torch.isnan(tensor).any(), f"{name} contains NaN"
            assert not torch.isinf(tensor).any(), f"{name} contains Inf"

        # Check expert indices are valid (0 to num_experts-1)
        assert (expert_indices >= 0).all() and (expert_indices < 4).all(), \
            f"expert_indices out of range [0, 3]: min={expert_indices.min()}, max={expert_indices.max()}"

        # Check expert weights sum to ~1 per token (softmax output)
        weight_sums = expert_weights.float().sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-2), \
            f"expert_weights do not sum to 1: {weight_sums}"

        print("mlp_gating: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"mlp_gating: TEST FAILED - {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
