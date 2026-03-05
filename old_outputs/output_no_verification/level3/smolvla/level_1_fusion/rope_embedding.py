"""
Component: Rotary Position Embedding (RoPE)
Abstraction Level: fusion
Parent: vlm_expert_transformer (layer)

Operations: frequency computation (arange, pow), sinusoidal rotation (sin, cos),
            split, element-wise multiply, concatenate

Input Shapes:
  - x: [B, S, H, D] float32 (H=num_heads, D=head_dim=64)
  - positions: [B, S] int64

Output Shapes:
  - rotated: [B, S, H, D] float32

Weight Shapes: (none - no learnable parameters)
"""
import torch
import torch.nn as nn

TEXT_HEAD_DIM = 64


class Model(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, max_wavelength=10_000):
        super().__init__()
        self.max_wavelength = max_wavelength

    def forward(self, x, positions):
        d_half = x.shape[-1] // 2
        device = x.device
        dtype = x.dtype
        x = x.to(torch.float32)

        freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
        timescale = self.max_wavelength ** freq_exponents
        radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
        radians = radians[..., None, :]

        sin = torch.sin(radians)
        cos = torch.cos(radians)

        x1, x2 = x.split(d_half, dim=-1)
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin
        return res.to(dtype)


def get_inputs():
    B, S, H, D = 1, 163, 15, TEXT_HEAD_DIM
    x = torch.randn(B, S, H, D)
    positions = torch.arange(S).unsqueeze(0)
    return [x, positions]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 163, 15, TEXT_HEAD_DIM)]

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
            print(f"Input shapes: x={inputs[0].shape}, positions={inputs[1].shape}")
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
