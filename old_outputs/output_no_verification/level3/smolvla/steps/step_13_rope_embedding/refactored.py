"""
Step 13 Refactored: rope_embedding - stateless positional encoding (no learnable parameters).
This is already at kernel level (basic trig operations + arithmetic).
The refactored version is identical to the original.
"""
import torch
import torch.nn as nn

TEXT_HEAD_DIM = 64

class RefactoredModel(nn.Module):
    """RoPE - already atomic (stateless computation)."""
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
    return [torch.randn(B, S, H, D), torch.arange(S).unsqueeze(0)]
def get_init_inputs():
    return []
def get_expected_output_shape():
    return [(1, 163, 15, TEXT_HEAD_DIM)]

if __name__ == "__main__":
    model = RefactoredModel(); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
