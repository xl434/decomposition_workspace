"""
Level 2 Layer: Patch Embedding
Full patch embedding pipeline: unfold + reshape + Linear + CLS token + positional embedding + dropout.
Fuses: level_1_fusion/patch_unfold_embed.py + level_1_fusion/cls_pos_dropout.py
Input: [batch_size, channels, image_size, image_size] = [2, 3, 16, 16]
Output: [batch_size, num_patches+1, dim] = [2, 17, 32]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, image_size=16, patch_size=4, dim=32, channels=3, emb_dropout=0.0):
        super(Model, self).__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2  # 16
        patch_dim = channels * patch_size ** 2  # 48

        self.patch_size = patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, img):
        p = self.patch_size
        # Unfold into patches and flatten
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(img.shape[0], -1, p * p * img.shape[1])
        # Project to embedding dim
        x = self.patch_to_embedding(x)
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embedding
        x = x + self.pos_embedding
        # Apply dropout
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.randn(2, 3, 16, 16)]


def get_init_inputs():
    return [16, 4, 32, 3, 0.0]  # image_size, patch_size, dim, channels, emb_dropout


def get_expected_output_shape():
    return (2, 17, 32)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"

    # Verify decomposition matches fused steps
    img = inputs[0]
    with torch.no_grad():
        # Step 1: patch_unfold_embed
        p = 4
        patches = img.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, 48)
        embedded = model.patch_to_embedding(patches)  # [2, 16, 32]
        # Step 2: cls_pos_dropout
        cls_tokens = model.cls_token.expand(2, -1, -1)
        with_cls = torch.cat((cls_tokens, embedded), dim=1)  # [2, 17, 32]
        with_pos = with_cls + model.pos_embedding  # [2, 17, 32]
        after_drop = model.dropout(with_pos)  # [2, 17, 32]
    assert torch.allclose(output, after_drop, atol=1e-6), "Layer vs decomposed mismatch"
    print(f"patch_embedding: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
