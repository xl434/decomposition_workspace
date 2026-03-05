"""
Component: Linear (fully connected) projection
Abstraction Level: kernel
Operations: matrix multiply + optional bias add

Input Shapes:
  - x: [B, S, in_features] float32

Output Shapes:
  - out: [B, S, out_features] float32

Weight Shapes:
  - weight: [out_features, in_features]
  - bias: [out_features] (optional)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Single Linear projection kernel."""
    def __init__(self, in_features=768, out_features=768, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    return [torch.randn(1, 64, 768)]

def get_init_inputs():
    return [768, 768, True]

def get_expected_output_shape():
    return [(1, 64, 768)]

def run_tests():
    try:
        model = Model(*get_init_inputs())
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
