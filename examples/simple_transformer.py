"""
Simple Transformer Block - Example for Decomposition

This is a self-contained transformer block suitable for testing
hierarchical decomposition. No external dependencies.

Expected decomposition:
- Level 2 (Layer): TransformerBlock
  - Level 1 (Fusion): Attention, MLP
    - Level 0 (Kernel): Linear, LayerNorm, Softmax, GELU, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Simple Transformer Block with:
    - Multi-head self-attention
    - Feed-forward network (MLP)
    - Residual connections
    - Layer normalization
    """

    def __init__(self, hidden_dim=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mlp_dim = hidden_dim * mlp_ratio

        # Attention components
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # MLP components
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, hidden_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch, seq_len, _ = x.shape

        # Self-attention block
        residual = x
        x = self.norm1(x)

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)

        # Reshape back
        x = x.transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        x = self.proj(x)
        x = x + residual

        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x + residual

        return x


def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    seq_len = 32
    hidden_dim = 768
    return [torch.randn(batch_size, seq_len, hidden_dim)]


def get_init_inputs():
    """Get initialization parameters."""
    return []  # Use defaults: hidden_dim=768, num_heads=12, mlp_ratio=4


def run_tests():
    """Verify this model executes correctly."""
    try:
        model = Model(*get_init_inputs())
        model.eval()

        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)

            # Validate
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            assert output.shape == inputs[0].shape, f"Shape mismatch: {output.shape} vs {inputs[0].shape}"

            print(f"Input shape: {inputs[0].shape}")
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
