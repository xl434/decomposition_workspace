"""
Level 1 Fusion: CLS Token Extraction + MLP Head
Extracts the CLS token then applies the classification MLP head.
Fuses: level_0_kernel/cls_token_extract.py + level_0_kernel/linear_head_up.py +
       level_0_kernel/gelu.py + level_0_kernel/dropout.py + level_0_kernel/linear_head_down.py
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, num_classes] = [2, 10]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32, mlp_dim=64, num_classes=10, dropout=0.0):
        super(Model, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x):
        # Extract CLS token (first position)
        cls_output = x[:, 0]
        # Apply MLP head
        return self.mlp_head(cls_output)


def get_inputs():
    return [torch.randn(2, 17, 32)]


def get_init_inputs():
    return [32, 64, 10, 0.0]  # dim, mlp_dim, num_classes, dropout


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

    # Verify decomposition: extract CLS then MLP separately
    x = inputs[0]
    with torch.no_grad():
        cls_out = x[:, 0]  # [2, 32]
        mlp_out = model.mlp_head(cls_out)  # [2, 10]
    assert torch.allclose(output, mlp_out, atol=1e-6), "Fusion vs decomposed mismatch"
    print(f"cls_extract_mlp: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
