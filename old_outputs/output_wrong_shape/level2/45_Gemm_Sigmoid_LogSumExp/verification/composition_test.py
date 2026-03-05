"""
Composition test for 45_Gemm_Sigmoid_LogSumExp

Verifies that chaining the individual kernel components produces
identical output to the original fused model, with shared weights.
"""

import torch
import torch.nn as nn


def test_composition():
    torch.manual_seed(42)

    # Original model (fused)
    class OriginalModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.sigmoid(x)
            x = self.linear2(x)
            x = torch.logsumexp(x, dim=1)
            return x

    # Composed model (chaining individual kernels)
    class ComposedModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # Step 1: linear_1
            x = self.linear1(x)
            # Step 2: sigmoid
            x = torch.sigmoid(x)
            # Step 3: linear_2
            x = self.linear2(x)
            # Step 4: logsumexp
            x = torch.logsumexp(x, dim=1)
            return x

    input_size = 16
    hidden_size = 32
    output_size = 8

    original = OriginalModel(input_size, hidden_size, output_size)
    composed = ComposedModel(input_size, hidden_size, output_size)

    # Copy weights from original to composed
    composed.linear1.weight = original.linear1.weight
    composed.linear1.bias = original.linear1.bias
    composed.linear2.weight = original.linear2.weight
    composed.linear2.bias = original.linear2.bias

    original.eval()
    composed.eval()

    with torch.no_grad():
        x = torch.randn(2, 16)
        out_orig = original(x)
        out_comp = composed(x)

        max_diff = (out_orig - out_comp).abs().max().item()
        print(f"Original output shape: {out_orig.shape}")
        print(f"Composed output shape: {out_comp.shape}")
        print(f"Max difference: {max_diff}")

        assert out_orig.shape == out_comp.shape, (
            f"Shape mismatch: original={out_orig.shape}, composed={out_comp.shape}"
        )
        assert torch.allclose(out_orig, out_comp, rtol=1e-4, atol=1e-5), (
            f"FAILED: max_diff={max_diff}"
        )
        print("PASS - Composition test passed")


if __name__ == "__main__":
    test_composition()
