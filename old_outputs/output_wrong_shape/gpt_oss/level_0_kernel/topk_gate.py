"""
Component: TopKGate
Level: 0 (Kernel)
Parent: MoELayer (Level 1)
Children: None (leaf kernel)
Operations: topk selection of experts, softmax over selected expert logits
Input Shapes:
    gate_logits: [16, 4] bfloat16 - gate scores for each expert (n_tokens, num_experts)
Output Shapes:
    expert_weights: [16, 2] bfloat16 - softmax-normalized weights for selected experts
    expert_indices: [16, 2] int64 - indices of selected experts
Weight Shapes: None (no learnable parameters; gate linear is a separate component)
"""

import sys
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, experts_per_token: int):
        super().__init__()
        self.experts_per_token = experts_per_token

    def forward(self, gate_logits):
        # gate_logits: [n_tokens, num_experts]
        experts = torch.topk(
            gate_logits, k=self.experts_per_token, dim=-1, sorted=True
        )
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        return expert_weights, expert_indices


def get_inputs():
    gate_logits = torch.randn(16, 4, dtype=torch.bfloat16)
    return [gate_logits]


def get_init_inputs():
    return [2]  # experts_per_token


def get_expected_output_shape():
    return [(16, 2), (16, 2)]


def run_tests():
    try:
        model = Model(*get_init_inputs())

        inputs = get_inputs()
        expert_weights, expert_indices = model(*inputs)

        expected_shapes = get_expected_output_shape()

        # Check expert_weights shape
        if tuple(expert_weights.shape) != expected_shapes[0]:
            print(
                f"FAIL: Expected expert_weights shape {expected_shapes[0]}, "
                f"got {tuple(expert_weights.shape)}"
            )
            return False

        # Check expert_indices shape
        if tuple(expert_indices.shape) != expected_shapes[1]:
            print(
                f"FAIL: Expected expert_indices shape {expected_shapes[1]}, "
                f"got {tuple(expert_indices.shape)}"
            )
            return False

        # Check expert_weights dtype
        if expert_weights.dtype != torch.bfloat16:
            print(
                f"FAIL: Expected expert_weights dtype bfloat16, got {expert_weights.dtype}"
            )
            return False

        # Check expert_indices dtype
        if expert_indices.dtype != torch.int64:
            print(
                f"FAIL: Expected expert_indices dtype int64, got {expert_indices.dtype}"
            )
            return False

        # Check for NaN in weights
        if torch.isnan(expert_weights).any():
            print("FAIL: expert_weights contains NaN")
            return False

        # Check for Inf in weights
        if torch.isinf(expert_weights).any():
            print("FAIL: expert_weights contains Inf")
            return False

        # Softmax weights should be non-negative and sum to ~1
        if (expert_weights < 0).any():
            print("FAIL: expert_weights contains negative values")
            return False

        row_sums = expert_weights.sum(dim=-1)
        if not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-2, rtol=1e-2
        ):
            print("FAIL: expert_weights rows do not sum to 1")
            return False

        # Expert indices should be in valid range [0, num_experts)
        if (expert_indices < 0).any() or (expert_indices >= 4).any():
            print("FAIL: expert_indices out of valid range [0, 4)")
            return False

        print("PASS: All tests passed for TopKGate")
        return True
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
