"""
Component: SiLU Linear MLP (2-layer MLP with SiLU activation)
Abstraction Level: fusion
Parent: action_time_mlp (layer)

Operations: Linear, SiLU activation, Linear

Input Shapes:
  - x: [B, 50, 1440] float32

Output Shapes:
  - out: [B, 50, 720] float32

Weight Shapes:
  - mlp_in: Linear(1440, 720)
  - mlp_out: Linear(720, 720)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_HIDDEN_SIZE = 720


class Model(nn.Module):
    """2-layer MLP with SiLU activation."""
    def __init__(self, in_features=EXPERT_HIDDEN_SIZE * 2, hidden_features=EXPERT_HIDDEN_SIZE,
                 out_features=EXPERT_HIDDEN_SIZE):
        super().__init__()
        self.mlp_in = nn.Linear(in_features, hidden_features)
        self.mlp_out = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.mlp_in(x)
        x = F.silu(x)
        x = self.mlp_out(x)
        return x


def get_inputs():
    return [torch.randn(1, 50, EXPERT_HIDDEN_SIZE * 2)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 50, EXPERT_HIDDEN_SIZE)]

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
            assert tuple(output.shape) == tuple(expected[0]), \
                f"Shape mismatch: {output.shape} vs {expected[0]}"
            print(f"Input shape: {inputs[0].shape}")
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
