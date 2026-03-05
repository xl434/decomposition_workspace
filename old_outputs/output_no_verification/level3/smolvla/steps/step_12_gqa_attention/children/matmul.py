"""
Component: Matrix Multiplication (batched)
Abstraction Level: kernel
Operations: torch.matmul (batched matrix multiply)

Input Shapes:
  - a: [B, H, M, K] float32
  - b: [B, H, K, N] float32

Output Shapes:
  - out: [B, H, M, N] float32

Weight Shapes: (none)
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Batched matrix multiplication kernel."""
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)


def get_inputs():
    return [torch.randn(1, 15, 163, 64), torch.randn(1, 15, 64, 163)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 15, 163, 163)]

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
