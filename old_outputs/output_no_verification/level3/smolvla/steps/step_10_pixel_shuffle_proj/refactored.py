"""
Step 10 Refactored: pixel_shuffle_proj decomposed into kernels.
Children: proj (linear kernel) - pixel shuffle is pure reshaping (data plumbing)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)
import linear as linear_mod

VISION_HIDDEN_SIZE = 768
SCALE_FACTOR = 4
CONNECTOR_INPUT_DIM = VISION_HIDDEN_SIZE * (SCALE_FACTOR ** 2)  # 12288
CONNECTOR_OUTPUT_DIM = 960
VISION_NUM_PATCHES = 1024

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = linear_mod.Model(CONNECTOR_INPUT_DIM, CONNECTOR_OUTPUT_DIM, bias=False)

    def forward(self, image_hidden_states):
        # Pixel shuffle (pure reshaping - allowed data plumbing)
        bsz, seq, embed_dim = image_hidden_states.size()
        height = width = int(seq ** 0.5)
        sf = SCALE_FACTOR
        x = image_hidden_states.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / sf), embed_dim * sf)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / sf), int(height / sf), embed_dim * (sf ** 2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (sf ** 2)), embed_dim * (sf ** 2))
        # Linear projection
        x = self.proj(x)
        return x

def get_inputs():
    return [torch.randn(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]
def get_init_inputs():
    return []
def get_expected_output_shape():
    return [(1, 64, CONNECTOR_OUTPUT_DIM)]

if __name__ == "__main__":
    model = RefactoredModel(); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
