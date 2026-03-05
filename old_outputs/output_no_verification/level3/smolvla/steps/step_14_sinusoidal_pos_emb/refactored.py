"""
Step 14 Refactored: sinusoidal_pos_emb - stateless positional encoding (no learnable parameters).
Already at kernel level (basic trig + arithmetic operations).
"""
import math
import torch
import torch.nn as nn

EXPERT_HIDDEN_SIZE = 720
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0

class RefactoredModel(nn.Module):
    """Sinusoidal positional embedding - already atomic."""
    def __init__(self, dimension=EXPERT_HIDDEN_SIZE, min_period=MIN_PERIOD, max_period=MAX_PERIOD):
        super().__init__()
        self.dimension = dimension
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, time):
        dimension = self.dimension
        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        if time.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
        device = time.device
        dtype = torch.float64 if device.type != "mps" else torch.float32
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * time[:, None]
        pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
        return pos_emb

def get_inputs():
    return [torch.rand(1) * 0.999 + 0.001]
def get_init_inputs():
    return []
def get_expected_output_shape():
    return [(1, EXPERT_HIDDEN_SIZE)]

if __name__ == "__main__":
    model = RefactoredModel(); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
