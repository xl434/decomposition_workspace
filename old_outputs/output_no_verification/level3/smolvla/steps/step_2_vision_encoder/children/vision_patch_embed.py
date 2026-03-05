"""
Component: Vision Patch Embedding + Position Embedding
Abstraction Level: fusion
Parent: vision_encoder (layer)

Operations: Conv2d patch embedding, flatten, transpose, position ID computation,
            Embedding lookup, addition

Input Shapes:
  - pixel_values: [B, 3, 512, 512] float32

Output Shapes:
  - embeddings: [B, 1024, 768] float32

Weight Shapes:
  - patch_embedding: Conv2d(3, 768, kernel_size=16, stride=16)
  - position_embedding: Embedding(1024, 768)
"""
import torch
import torch.nn as nn

VISION_HIDDEN_SIZE = 768
VISION_IMAGE_SIZE = 512
VISION_PATCH_SIZE = 16
VISION_NUM_CHANNELS = 3
VISION_NUM_PATCHES_PER_SIDE = VISION_IMAGE_SIZE // VISION_PATCH_SIZE  # 32
VISION_NUM_PATCHES = VISION_NUM_PATCHES_PER_SIDE ** 2  # 1024


class Model(nn.Module):
    """Vision patch + position embedding."""
    def __init__(self):
        super().__init__()
        self.patch_size = VISION_PATCH_SIZE
        self.num_patches_per_side = VISION_NUM_PATCHES_PER_SIDE
        self.num_patches = VISION_NUM_PATCHES

        self.patch_embedding = nn.Conv2d(
            in_channels=VISION_NUM_CHANNELS,
            out_channels=VISION_HIDDEN_SIZE,
            kernel_size=VISION_PATCH_SIZE,
            stride=VISION_PATCH_SIZE,
        )
        self.position_embedding = nn.Embedding(VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)

    def forward(self, pixel_values):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size

        patch_attention_mask = torch.ones(
            (batch_size, max_nb_patches_h, max_nb_patches_w),
            dtype=torch.bool, device=pixel_values.device
        )

        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
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
            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


def get_inputs():
    return [torch.randn(1, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]

def run_tests():
    try:
        model = Model()
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
            print(f"Input shape: {inputs[0].shape}")
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
