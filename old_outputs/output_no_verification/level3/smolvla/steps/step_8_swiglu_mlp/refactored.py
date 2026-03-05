"""
Step 8 Refactored: swiglu_mlp decomposed into kernels.
Children: gate_proj (linear), activation (silu), up_proj (linear), down_proj (linear)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)
import linear as linear_mod
import silu as silu_mod

TEXT_HIDDEN_SIZE = 960
TEXT_INTERMEDIATE_SIZE = 2560

class RefactoredModel(nn.Module):
    def __init__(self, hidden_size=TEXT_HIDDEN_SIZE, intermediate_size=TEXT_INTERMEDIATE_SIZE):
        super().__init__()
        self.gate_proj = linear_mod.Model(hidden_size, intermediate_size, bias=False)
        self.activation = silu_mod.Model()
        self.up_proj = linear_mod.Model(hidden_size, intermediate_size, bias=False)
        self.down_proj = linear_mod.Model(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = self.activation(gate)
        up = self.up_proj(x)
        return self.down_proj(gate * up)

def get_inputs():
    return [torch.randn(1, 113, TEXT_HIDDEN_SIZE)]
def get_init_inputs():
    return []
def get_expected_output_shape():
    return [(1, 113, TEXT_HIDDEN_SIZE)]

if __name__ == "__main__":
    model = RefactoredModel(); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
