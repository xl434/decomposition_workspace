"""
Level 0 Kernel: Patch Unfold
Unfolds an image into non-overlapping patches and reshapes to flat patch vectors.
Input: [batch_size, channels, image_size, image_size] = [2, 3, 16, 16]
Output: [batch_size, num_patches, patch_dim] = [2, 16, 48]
No learnable weights.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, patch_size=4, channels=3):
        super(Model, self).__init__()
        self.patch_size = patch_size
        self.channels = channels

    def forward(self, img):
        p = self.patch_size
        # unfold height, then width, then reshape to (batch, num_patches, patch_dim)
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(img.shape[0], -1, p * p * img.shape[1])
        return x


def get_inputs():
    return [torch.randn(2, 3, 16, 16)]


def get_init_inputs():
    return [4, 3]  # patch_size, channels


def get_expected_output_shape():
    return (2, 16, 48)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    print(f"patch_unfold: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
