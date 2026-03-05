"""
Component: RMS Normalization
Abstraction Level: kernel (L0)
Parent: gpt_oss (L3)
Children: None (leaf)

Operations: RMSNorm with learnable scale parameter

Input Shapes:
  - x: [16, 128] dtype=bfloat16

Output Shapes:
  - output: [16, 128] dtype=bfloat16

Weight Shapes:
  - scale: [128] dtype=float32
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """RMS Normalization. Extracted from: Transformer"""

    def __init__(self, num_features=128, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, dtype=torch.float32))

    def forward(self, x):
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def get_inputs():
    return [torch.randn(16, 128, dtype=torch.bfloat16)]


def get_init_inputs():
    return [128, 1e-6]


def get_expected_output_shape():
    return [(16, 128)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape] if not isinstance(output, tuple) else [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Output {i} shape mismatch: got {actual}, expected {expected}"
            assert output.dtype == torch.bfloat16, f"Dtype mismatch: {output.dtype} vs bfloat16"
            # Validate scale parameter dtype
            assert model.scale.dtype == torch.float32, f"Scale dtype mismatch: {model.scale.dtype} vs float32"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
