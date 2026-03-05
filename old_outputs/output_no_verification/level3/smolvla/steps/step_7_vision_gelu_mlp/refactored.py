"""
Step 7 Refactored: vision_gelu_mlp decomposed into kernels.
Children: fc1 (linear), activation (gelu), fc2 (linear)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)
import linear as linear_mod
import gelu as gelu_mod

VISION_HIDDEN_SIZE = 768
VISION_INTERMEDIATE_SIZE = 3072
VISION_NUM_PATCHES = 1024

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = linear_mod.Model(VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE)
        self.activation = gelu_mod.Model()
        self.fc2 = linear_mod.Model(VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

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
