"""
Component: Causal + Sliding Window Attention Mask
Abstraction Level: kernel (L0)
Parent: gpt_oss (L3)
Children: None (leaf)

Operations: Create causal mask with optional sliding window

Input Shapes:
  - x: [16] (dummy tensor to convey sequence length)

Output Shapes:
  - mask: [16, 16] dtype=float32

Config:
  - sliding_window = 8
  - Causal mask: upper triangular -inf
  - Sliding window: lower triangular -inf beyond window size
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """Create causal + sliding window attention mask. Extracted from: Transformer"""

    def __init__(self, sliding_window=8):
        super().__init__()
        self.sliding_window = sliding_window

    def forward(self, x):
        n_tokens = x.shape[0]
        mask = torch.triu(x.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
        if self.sliding_window > 0:
            mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-self.sliding_window)
        return mask


def get_inputs():
    return [torch.zeros(16)]


def get_init_inputs():
    return [8]


def get_expected_output_shape():
    return [(16, 16)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape] if not isinstance(output, tuple) else [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Output {i} shape mismatch: got {actual}, expected {expected}"
            assert output.dtype == torch.float32, f"Dtype mismatch: {output.dtype} vs float32"
            # Validate causal structure: upper triangle should be -inf
            for row in range(16):
                for col in range(row + 1, 16):
                    assert output[row, col] == -float("inf"), f"Expected -inf at [{row},{col}], got {output[row, col]}"
            # Validate sliding window: positions beyond window should be -inf
            for row in range(16):
                for col in range(0, max(0, row - 8 + 1)):
                    assert output[row, col] == -float("inf"), f"Expected -inf at [{row},{col}] (sliding window), got {output[row, col]}"
            # Validate diagonal is 0
            for i in range(16):
                assert output[i, i] == 0.0, f"Expected 0.0 at diagonal [{i},{i}], got {output[i, i]}"
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
