"""
Composition test for 38_LSTMBidirectional.

Verifies that composing the 3 Level 0 kernels (lstm, slice_last, linear)
in sequence produces output identical to the Level 1 fusion and to the
original model.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from level_0_kernel.lstm import Model as LSTMKernel
from level_0_kernel.slice_last import Model as SliceLastKernel
from level_0_kernel.linear import Model as LinearKernel
from level_1_fusion.lstm_slice_fc import Model as FusionModel, get_inputs, get_init_inputs


class OriginalModel(nn.Module):
    """Original model from KernelBench level3/38_LSTMBidirectional."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(OriginalModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, h0, c0):
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def run_tests():
    """Verify composed kernels match the original model and the fusion."""
    torch.manual_seed(42)

    # Build original model
    original = OriginalModel(input_size=8, hidden_size=16, num_layers=2, output_size=4)
    original.eval()

    # Build fusion model and share weights with original
    fusion = FusionModel(*get_init_inputs())
    fusion.load_state_dict(original.state_dict())
    fusion.eval()

    # Build individual kernels and share weights
    lstm_kernel = LSTMKernel(input_size=8, hidden_size=16, num_layers=2)
    slice_kernel = SliceLastKernel()
    linear_kernel = LinearKernel(in_features=32, out_features=4)

    # Copy LSTM weights from original
    lstm_state = {k.replace("lstm.", ""): v for k, v in original.state_dict().items() if k.startswith("lstm.")}
    lstm_kernel.lstm.load_state_dict(lstm_state)
    lstm_kernel.eval()

    # Copy FC weights from original
    fc_state = {k.replace("fc.", ""): v for k, v in original.state_dict().items() if k.startswith("fc.")}
    linear_kernel.fc.load_state_dict(fc_state)
    linear_kernel.eval()

    # Generate inputs
    inputs = get_inputs()
    x, h0, c0 = inputs

    with torch.no_grad():
        # Run original
        out_original = original(x, h0, c0)

        # Run fusion
        out_fusion = fusion(x, h0, c0)

        # Run composed kernels step by step
        step1 = lstm_kernel(x, h0, c0)       # [2, 4, 32]
        step2 = slice_kernel(step1)            # [2, 32]
        step3 = linear_kernel(step2)           # [2, 4]
        out_composed = step3

    # Compare fusion vs original
    match_fusion = torch.allclose(out_original, out_fusion, rtol=1e-4, atol=1e-5)
    diff_fusion = (out_original - out_fusion).abs().max().item()
    assert match_fusion, f"Fusion vs original mismatch! max diff = {diff_fusion}"
    print(f"[PASS] fusion vs original: match (max diff = {diff_fusion:.2e})")

    # Compare composed vs original
    match_composed = torch.allclose(out_original, out_composed, rtol=1e-4, atol=1e-5)
    diff_composed = (out_original - out_composed).abs().max().item()
    assert match_composed, f"Composed vs original mismatch! max diff = {diff_composed}"
    print(f"[PASS] composed kernels vs original: match (max diff = {diff_composed:.2e})")

    # Verify intermediate shapes
    assert step1.shape == (2, 4, 32), f"LSTM output shape wrong: {step1.shape}"
    assert step2.shape == (2, 32), f"Slice output shape wrong: {step2.shape}"
    assert step3.shape == (2, 4), f"Linear output shape wrong: {step3.shape}"
    print(f"[PASS] intermediate shapes: LSTM={step1.shape}, slice={step2.shape}, FC={step3.shape}")

    print(f"       final output shape: {out_original.shape}")
    return True


if __name__ == "__main__":
    run_tests()
