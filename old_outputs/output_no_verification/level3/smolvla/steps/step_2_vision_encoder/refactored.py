"""
Step 2 Refactored: vision_encoder decomposed into fusion-level children.

Children:
  - embeddings: VisionPatchEmbed (L1 fusion) - patch + position embedding
  - attention_layers: 12x VisionSDPA (L1 fusion) - multi-head self-attention
  - mlp_layers: 12x VisionGELU_MLP (L1 fusion) - GELU MLP
  - layer_norms_1: 12x VisionLayerNorm (L1 fusion) - pre-attention LayerNorm
  - layer_norms_2: 12x VisionLayerNorm (L1 fusion) - pre-MLP LayerNorm
  - post_layernorm: VisionLayerNorm (L1 fusion) - final LayerNorm
  - connector: PixelShuffleProj (L1 fusion) - pixel shuffle + projection

forward() only does data plumbing: residuals (+), looping.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)

import vision_patch_embed as vpe_mod
import vision_sdpa as vsdpa_mod
import vision_gelu_mlp as vmlp_mod
import vision_layernorm as vln_mod
import pixel_shuffle_proj as psp_mod

VISION_NUM_LAYERS = 12
VISION_IMAGE_SIZE = 512
VISION_HIDDEN_SIZE = 768


class RefactoredModel(nn.Module):
    """Vision encoder decomposed into fusion-level children."""

    def __init__(self):
        super().__init__()
        self.embeddings = vpe_mod.Model()
        self.attention_layers = nn.ModuleList([vsdpa_mod.Model() for _ in range(VISION_NUM_LAYERS)])
        self.mlp_layers = nn.ModuleList([vmlp_mod.Model() for _ in range(VISION_NUM_LAYERS)])
        self.layer_norms_1 = nn.ModuleList([vln_mod.Model() for _ in range(VISION_NUM_LAYERS)])
        self.layer_norms_2 = nn.ModuleList([vln_mod.Model() for _ in range(VISION_NUM_LAYERS)])
        self.post_layernorm = vln_mod.Model()
        self.connector = psp_mod.Model()

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)

        for i in range(VISION_NUM_LAYERS):
            residual = hidden_states
            hidden_states = self.layer_norms_1[i](hidden_states)
            hidden_states = self.attention_layers[i](hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norms_2[i](hidden_states)
            hidden_states = self.mlp_layers[i](hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.post_layernorm(hidden_states)
        image_embeddings = self.connector(hidden_states)
        return image_embeddings


def get_inputs():
    return [torch.randn(1, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 64, 960)]

def run_tests():
    try:
        model = RefactoredModel()
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
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
