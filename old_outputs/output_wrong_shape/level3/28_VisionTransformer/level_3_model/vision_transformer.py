"""
Level 3 Model: Vision Transformer (ViT)
Full Vision Transformer model combining patch embedding, transformer encoder,
and classification head.
Fuses: level_2_layer/patch_embedding.py + level_2_layer/transformer_encoder.py +
       level_2_layer/classification_head.py
Input: [batch_size, channels, image_size, image_size] = [2, 3, 16, 16]
Output: [batch_size, num_classes] = [2, 10]

Test dimensions:
  image_size=16, patch_size=4, num_classes=10, dim=32, depth=2,
  heads=4, mlp_dim=64, channels=3, dropout=0.0, emb_dropout=0.0
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(Model, self).__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout
            ),
            num_layers=depth,
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, img):
        p = self.patch_size
        # Patch embedding
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(
            img.shape[0], -1, p * p * img.shape[1]
        )
        x = self.patch_to_embedding(x)
        # CLS token + positional embedding + dropout
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        # Transformer encoder
        x = self.transformer(x)
        # Classification head
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def get_inputs():
    return [torch.randn(2, 3, 16, 16)]


def get_init_inputs():
    return [16, 4, 10, 32, 2, 4, 64, 3, 0.0, 0.0]
    # image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout


def get_expected_output_shape():
    return (2, 10)


def run_tests():
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"

    # Verify intermediate shapes
    img = inputs[0]
    p = model.patch_size
    with torch.no_grad():
        # Step 1: Patch unfold + embed
        patches = img.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, p * p * 3)
        assert patches.shape == (2, 16, 48), f"Patches shape: {patches.shape}"
        embedded = model.patch_to_embedding(patches)
        assert embedded.shape == (2, 16, 32), f"Embedded shape: {embedded.shape}"

        # Step 2: CLS + pos + dropout
        cls_tokens = model.cls_token.expand(2, -1, -1)
        with_cls = torch.cat((cls_tokens, embedded), dim=1)
        assert with_cls.shape == (2, 17, 32), f"With CLS shape: {with_cls.shape}"
        with_pos = with_cls + model.pos_embedding
        after_drop = model.dropout(with_pos)
        assert after_drop.shape == (2, 17, 32), f"After dropout shape: {after_drop.shape}"

        # Step 3: Transformer
        transformed = model.transformer(after_drop)
        assert transformed.shape == (2, 17, 32), f"Transformed shape: {transformed.shape}"

        # Step 4: Classification
        cls_out = model.to_cls_token(transformed[:, 0])
        assert cls_out.shape == (2, 32), f"CLS output shape: {cls_out.shape}"
        final = model.mlp_head(cls_out)
        assert final.shape == (2, 10), f"Final shape: {final.shape}"

    # Verify end-to-end match
    assert torch.allclose(output, final, atol=1e-5), "Full model vs step-by-step mismatch"
    print(f"vision_transformer: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
