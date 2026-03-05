"""
Step 11 Refactored: rms_norm - already atomic (single normalization operation).
This is effectively an L0 kernel. The refactored version is identical.
Children: weight parameter only (no further decomposition possible)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

TEXT_RMS_NORM_EPS = 1e-5

class RefactoredModel(nn.Module):
    """RMS Normalization - atomic kernel."""
    def __init__(self, hidden_size=960, eps=TEXT_RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

def get_inputs():
    return [torch.randn(1, 113, 960)]
def get_init_inputs():
    return [960]
def get_expected_output_shape():
    return [(1, 113, 960)]

if __name__ == "__main__":
    model = RefactoredModel(*get_init_inputs()); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
