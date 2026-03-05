"""
Level 0 Kernel: AdaptiveAvgPool1d
AdaptiveAvgPool1d(1) for global average pooling.

Input: [2, 128, 1] -> Output: [2, 128, 1]

In the full model, input is actually [B, 128, L] where L is the sequence length
from the last stage. After the last stage with our test dims, L=1.
This pools over the spatial/sequence dimension to produce a single value per channel.

For a more general test, we also test with L>1.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """AdaptiveAvgPool1d(1) for global average pooling.

    Input: [B, 128, L]  (L can be any length)
    Output: [B, 128, 1]
    """
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.avgpool(x)


def get_inputs():
    """Return sample inputs.
    Using L=1 to match actual model flow, but also test with L>1.
    """
    return [torch.randn(2, 128, 1)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 128, 1)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 0: AdaptiveAvgPool1d")
    print("=" * 60)

    model = Model()
    model.eval()

    inputs = get_inputs()
    expected_shapes = get_expected_output_shape()

    with torch.no_grad():
        output = model(*inputs)

    assert output.shape == expected_shapes[0], \
        f"Shape mismatch: {output.shape} vs {expected_shapes[0]}"
    print(f"[PASS] Output shape: {output.shape}")

    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("[PASS] No NaN/Inf in output")

    # Test with L>1 to verify pooling behavior
    x_multi = torch.randn(2, 128, 4)
    out_multi = model(x_multi)
    assert out_multi.shape == (2, 128, 1), \
        f"Multi-length shape mismatch: {out_multi.shape}"
    print(f"[PASS] Multi-length pooling: [2,128,4] -> {out_multi.shape}")

    # Verify pooling is average
    manual_avg = x_multi.mean(dim=-1, keepdim=True)
    assert torch.allclose(out_multi, manual_avg, atol=1e-6), \
        "Pooling should be average of spatial dim"
    print("[PASS] Average pooling verified")

    print("\n[PASS] All AdaptiveAvgPool1d tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
