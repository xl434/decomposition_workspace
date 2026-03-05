"""
Step 9 Refactored: vision_patch_embed decomposed into kernels.
Children: patch_conv (conv2d), position_emb (embedding)
forward() does: conv2d, flatten, transpose, position ID computation, embedding lookup, addition
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)
import conv2d as conv2d_mod
import embedding as emb_mod

VISION_HIDDEN_SIZE = 768
VISION_IMAGE_SIZE = 512
VISION_PATCH_SIZE = 16
VISION_NUM_CHANNELS = 3
VISION_NUM_PATCHES_PER_SIDE = VISION_IMAGE_SIZE // VISION_PATCH_SIZE
VISION_NUM_PATCHES = VISION_NUM_PATCHES_PER_SIDE ** 2

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_conv = conv2d_mod.Model(VISION_NUM_CHANNELS, VISION_HIDDEN_SIZE,
                                            VISION_PATCH_SIZE, VISION_PATCH_SIZE)
        self.position_emb = emb_mod.Model(VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)

    def forward(self, pixel_values):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        patch_embeds = self.patch_conv(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h = max_im_h // VISION_PATCH_SIZE
        max_nb_patches_w = max_im_w // VISION_PATCH_SIZE

        patch_attention_mask = torch.ones(
            (batch_size, max_nb_patches_h, max_nb_patches_w),
            dtype=torch.bool, device=pixel_values.device
        )

        boundaries = torch.arange(1 / VISION_NUM_PATCHES_PER_SIDE, 1.0, 1 / VISION_NUM_PATCHES_PER_SIDE)
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()
            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)
            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
            pos_ids = (bucket_coords_h[:, None] * VISION_NUM_PATCHES_PER_SIDE + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(pixel_values.device)
        pos_emb = self.position_emb(position_ids)
        embeddings = embeddings + pos_emb
        return embeddings

def get_inputs():
    return [torch.randn(1, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)]
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
