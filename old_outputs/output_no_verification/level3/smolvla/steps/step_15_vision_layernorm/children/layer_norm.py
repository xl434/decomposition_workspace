"""
Component: Layer Normalization
Abstraction Level: kernel
Operations: nn.LayerNorm

Input Shapes:
  - x: [B, S, D] float32

Output Shapes:
  - out: [B, S, D] float32

Weight Shapes:
  - weight: [D]
  - bias: [D]
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Layer Normalization kernel."""
    def __init__(self, normalized_shape=768, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.norm(x)


def get_inputs():
    return [torch.randn(1, 1024, 768)]

def get_init_inputs():
    return [768, 1e-6]

def get_expected_output_shape():
    return [(1, 1024, 768)]

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
