"""
Composition test for 41_GRUBidirectional.

Since this is a Level 0 kernel (single operation), the composition test
simply verifies that the kernel file runs correctly and matches the
original model's output exactly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from level_0_kernel.gru_bidirectional import Model as KernelModel, get_inputs, get_init_inputs


class OriginalModel(nn.Module):
    """Original model from KernelBench level3/41_GRUBidirectional."""

    def __init__(self, input_size, hidden_size, num_layers=2, bias=True, batch_first=False):
        super(OriginalModel, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            bias=bias, batch_first=batch_first,
            dropout=0, bidirectional=True
        )

    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return output


def run_tests():
    """Verify kernel output matches original model output exactly."""
    torch.manual_seed(42)

    # Build original model
    init_args = get_init_inputs()
    original = OriginalModel(*init_args)
    original.eval()

    # Build kernel model and share weights
    kernel = KernelModel(*init_args)
    kernel.load_state_dict(original.state_dict())
    kernel.eval()

    # Run both with same inputs
    inputs = get_inputs()
    with torch.no_grad():
        out_original = original(*inputs)
        out_kernel = kernel(*inputs)

    # Compare
    match = torch.allclose(out_original, out_kernel, rtol=1e-4, atol=1e-5)
    max_diff = (out_original - out_kernel).abs().max().item()
    assert match, f"Output mismatch! max diff = {max_diff}"
    print(f"[PASS] composition_test: outputs match (max diff = {max_diff:.2e})")
    print(f"       original shape: {out_original.shape}")
    print(f"       kernel shape:   {out_kernel.shape}")
    return True


if __name__ == "__main__":
    run_tests()
