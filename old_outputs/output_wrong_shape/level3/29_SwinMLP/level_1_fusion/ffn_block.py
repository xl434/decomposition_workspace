"""
Level 1 Fusion: FFN Block
LayerNorm + MLP (Linear + GELU + Linear) with residual connection.

Input: [2, 64, 16] -> Output: [2, 64, 16]

Params:
  - LayerNorm(16)
  - Linear(16, 32)  (hidden_features = dim * mlp_ratio = 16 * 2.0 = 32)
  - GELU
  - Linear(32, 16)
  - Residual connection: output = x + mlp(norm(x))
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """FFN Block: LayerNorm + MLP (Linear -> GELU -> Linear) with residual.

    Input: [B, 64, 16]
    Output: [B, 64, 16]
    """
    def __init__(self):
        super().__init__()
        dim = 16
        mlp_ratio = 2.0
        hidden_dim = int(dim * mlp_ratio)  # 32

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # x: [B, 64, 16]
        residual = x
        x = self.norm(x)          # [B, 64, 16]
        x = self.fc1(x)           # [B, 64, 32]
        x = self.act(x)           # [B, 64, 32]
        x = self.fc2(x)           # [B, 64, 16]
        x = residual + x          # [B, 64, 16]
        return x


def get_inputs():
    """Return sample inputs."""
    return [torch.randn(2, 64, 16)]


def get_init_inputs():
    """Return inputs for model initialization."""
    return []


def get_expected_output_shape():
    """Return expected output shapes."""
    return [(2, 64, 16)]


def run_tests():
    """Run validation tests."""
    print("=" * 60)
    print("Testing Level 1: FFN Block")
    print("=" * 60)

    torch.manual_seed(42)
    model = Model()
    model.eval()

    inputs = get_inputs()
    expected_shapes = get_expected_output_shape()

    with torch.no_grad():
        output = model(*inputs)

    # Shape test
    assert output.shape == expected_shapes[0], \
        f"Shape mismatch: {output.shape} vs {expected_shapes[0]}"
    print(f"[PASS] Output shape: {output.shape}")

    # No NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("[PASS] No NaN/Inf in output")

    # Verify residual connection
    x = get_inputs()[0]
    with torch.no_grad():
        normed = model.norm(x)
        fc1_out = model.fc1(normed)
        print(f"[INFO] After fc1: {fc1_out.shape}")          # [2, 64, 32]
        act_out = model.act(fc1_out)
        fc2_out = model.fc2(act_out)
        print(f"[INFO] After fc2: {fc2_out.shape}")          # [2, 64, 16]
        manual_out = x + fc2_out
        assert torch.allclose(output, manual_out, atol=1e-6)
    print("[PASS] Residual connection verified")

    print("\n[PASS] All FFN Block tests passed!")
    return True


if __name__ == "__main__":
    run_tests()
