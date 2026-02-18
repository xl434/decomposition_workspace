"""
Attention Block - Example for Decomposition

Multi-head self-attention block.

Expected decomposition:
- Level 1 (Fusion): Attention Block
  - Level 0 (Kernel): Linear (QKV), MatMul, Softmax, MatMul, Linear (Proj)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Multi-head self-attention block.
    """

    def __init__(self, hidden_dim=768, num_heads=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)

        # Output projection
        x = x.transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        x = self.proj(x)

        return x


def get_inputs():
    """Generate test inputs."""
    return [torch.randn(2, 32, 768)]


def get_init_inputs():
    """Get initialization parameters."""
    return []


def run_tests():
    """Verify this model executes correctly."""
    try:
        model = Model(*get_init_inputs())
        model.eval()

        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)

            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert output.shape == inputs[0].shape, f"Shape mismatch"

            print(f"Input shape: {inputs[0].shape}")
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
