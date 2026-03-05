"""
Level 1 Fusion: Patch Unfold + Linear Embedding
Combines image unfolding into patches with linear projection to embedding dim.
Fuses: level_0_kernel/patch_unfold.py + level_0_kernel/linear_embed.py
Input: [batch_size, channels, image_size, image_size] = [2, 3, 16, 16]
Output: [batch_size, num_patches, dim] = [2, 16, 32]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, patch_size=4, channels=3, dim=32):
        super(Model, self).__init__()
        self.patch_size = patch_size
        patch_dim = channels * patch_size ** 2  # 3 * 16 = 48
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

    def forward(self, img):
        p = self.patch_size
        # Unfold into patches and flatten
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(img.shape[0], -1, p * p * img.shape[1])
        # Project to embedding dim
        x = self.patch_to_embedding(x)
        return x


def get_inputs():
    return [torch.randn(2, 3, 16, 16)]


def get_init_inputs():
    return [4, 3, 32]  # patch_size, channels, dim


def get_expected_output_shape():
    return (2, 16, 32)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"

    # Verify decomposition: run unfold then linear separately
    img = inputs[0]
    p = 4
    with torch.no_grad():
        patches = img.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, 48)
        embedded = model.patch_to_embedding(patches)
    assert torch.allclose(output, embedded, atol=1e-6), "Fusion vs decomposed mismatch"
    print(f"patch_unfold_embed: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
