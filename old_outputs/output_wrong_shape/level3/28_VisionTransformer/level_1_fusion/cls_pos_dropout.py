"""
Level 1 Fusion: CLS Token + Positional Embedding + Dropout
Prepends CLS token, adds positional embeddings, then applies dropout.
Fuses: level_0_kernel/cls_token_cat.py + level_0_kernel/pos_embedding_add.py + level_0_kernel/dropout.py
Input: [batch_size, num_patches, dim] = [2, 16, 32]
Output: [batch_size, num_patches+1, dim] = [2, 17, 32]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_patches=16, dim=32, emb_dropout=0.0):
        super(Model, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embedding
        x = x + self.pos_embedding
        # Apply dropout
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.randn(2, 16, 32)]


def get_init_inputs():
    return [16, 32, 0.0]  # num_patches, dim, emb_dropout


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

    # Verify decomposition: run each step separately
    x = inputs[0]
    with torch.no_grad():
        cls_tokens = model.cls_token.expand(2, -1, -1)
        step1 = torch.cat((cls_tokens, x), dim=1)  # [2, 17, 32]
        step2 = step1 + model.pos_embedding  # [2, 17, 32]
        step3 = model.dropout(step2)  # [2, 17, 32]
    assert torch.allclose(output, step3, atol=1e-6), "Fusion vs decomposed mismatch"
    print(f"cls_pos_dropout: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
