"""
Level 2 Layer: Classification Head
Extracts CLS token and applies MLP head for classification.
Fuses: level_1_fusion/cls_extract_mlp.py
  (which itself fuses: cls_token_extract + linear_head_up + gelu + dropout + linear_head_down)
Input: [batch_size, seq_len, dim] = [2, 17, 32]
Output: [batch_size, num_classes] = [2, 10]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=32, mlp_dim=64, num_classes=10, dropout=0.0):
        super(Model, self).__init__()
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, x):
        # Extract CLS token via Identity (matching original model)
        cls_output = self.to_cls_token(x[:, 0])
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

    # Verify decomposition
    x = inputs[0]
    with torch.no_grad():
        cls_out = x[:, 0]  # [2, 32]
        step1 = model.mlp_head[0](cls_out)  # Linear(32,64) -> [2, 64]
        step2 = model.mlp_head[1](step1)  # GELU -> [2, 64]
        step3 = model.mlp_head[2](step2)  # Dropout -> [2, 64]
        step4 = model.mlp_head[3](step3)  # Linear(64,10) -> [2, 10]
    assert torch.allclose(output, step4, atol=1e-6), "Layer vs decomposed mismatch"
    print(f"classification_head: output shape {output.shape} - PASS")


if __name__ == "__main__":
    run_tests()
