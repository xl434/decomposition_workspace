"""
Step 15 Refactored: vision_layernorm decomposed into kernel.
Children: norm (layer_norm kernel)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)
import layer_norm as ln_mod

VISION_HIDDEN_SIZE = 768
VISION_LAYER_NORM_EPS = 1e-6
VISION_NUM_PATCHES = 1024

class RefactoredModel(nn.Module):
    def __init__(self, hidden_size=VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS):
        super().__init__()
        self.norm = ln_mod.Model(hidden_size, eps)

    def forward(self, hidden_states):
        return self.norm(hidden_states)

def get_inputs():
    return [torch.randn(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]
def get_init_inputs():
    return []
def get_expected_output_shape():
    return [(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]

if __name__ == "__main__":
    model = RefactoredModel(); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
