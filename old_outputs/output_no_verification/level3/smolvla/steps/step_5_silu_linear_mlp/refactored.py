"""
Step 5 Refactored: silu_linear_mlp decomposed into kernels.
Children: mlp_in (linear), activation (silu), mlp_out (linear)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)
import linear as linear_mod
import silu as silu_mod

EXPERT_HIDDEN_SIZE = 720

class RefactoredModel(nn.Module):
    def __init__(self, in_features=EXPERT_HIDDEN_SIZE * 2, hidden_features=EXPERT_HIDDEN_SIZE,
                 out_features=EXPERT_HIDDEN_SIZE):
        super().__init__()
        self.mlp_in = linear_mod.Model(in_features, hidden_features)
        self.activation = silu_mod.Model()
        self.mlp_out = linear_mod.Model(hidden_features, out_features)

    def forward(self, x):
        x = self.mlp_in(x)
        x = self.activation(x)
        x = self.mlp_out(x)
        return x

def get_inputs():
    return [torch.randn(1, 50, EXPERT_HIDDEN_SIZE * 2)]
def get_init_inputs():
    return []
def get_expected_output_shape():
    return [(1, 50, EXPERT_HIDDEN_SIZE)]

if __name__ == "__main__":
    model = RefactoredModel(); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
