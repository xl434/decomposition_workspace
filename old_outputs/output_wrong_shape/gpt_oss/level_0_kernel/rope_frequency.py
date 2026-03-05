"""
Component: YaRN Rotary Frequency Computation
Abstraction Level: kernel (L0)
Parent: gpt_oss (L3)
Children: None (leaf)

Operations: Compute YaRN-scaled rotary position embedding frequencies (cos, sin)

Input Shapes:
  - x: [16] (dummy tensor to convey sequence length)

Output Shapes:
  - cos: [16, 16] dtype=float32 (seq_len, head_dim // 2)
  - sin: [16, 16] dtype=float32 (seq_len, head_dim // 2)

Config:
  - head_dim = 32
  - base = 150000.0
  - scaling_factor = 32.0
  - ntk_alpha = 1.0
  - ntk_beta = 32.0
  - initial_context_length = 256
"""

import math
import torch
import torch.nn as nn


class Model(nn.Module):
    """Compute YaRN rotary frequencies (cos, sin). Extracted from: Transformer"""

    def __init__(self, head_dim=32, base=150000.0, scaling_factor=32.0,
                 ntk_alpha=1.0, ntk_beta=32.0, initial_context_length=256):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.initial_context_length = initial_context_length

    def forward(self, x):
        num_tokens = x.shape[0]
        # Compute inv_freq with YaRN scaling
        freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        concentration = 0.1 * math.log(self.scaling_factor) + 1.0
        d_half = self.head_dim / 2
        low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
        high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
        interpolation = 1.0 / (self.scaling_factor * freq)
        extrapolation = 1.0 / freq
        ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
        mask = 1 - ramp.clamp(0, 1)
        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        # Compute cos, sin
        t = torch.arange(num_tokens, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin


def get_inputs():
    return [torch.zeros(16)]


def get_init_inputs():
    return [32, 150000.0, 32.0, 1.0, 32.0, 256]


def get_expected_output_shape():
    return [(16, 16), (16, 16)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert isinstance(output, tuple), f"Expected tuple output, got {type(output)}"
            assert len(output) == 2, f"Expected 2 outputs, got {len(output)}"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Output {i} shape mismatch: got {actual}, expected {expected}"
            for i, o in enumerate(output):
                assert not torch.isnan(o).any(), f"Output {i} contains NaN"
                assert not torch.isinf(o).any(), f"Output {i} contains Inf"
                assert o.dtype == torch.float32, f"Output {i} dtype mismatch: {o.dtype} vs float32"
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
