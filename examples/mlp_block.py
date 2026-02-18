"""
MLP Block - Simple Example for Decomposition

This is a simple feed-forward network for testing decomposition.

Expected decomposition:
- Level 1 (Fusion): MLP Block
  - Level 0 (Kernel): Linear, GELU, Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Simple MLP block: Linear → GELU → Linear
    """

    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Output tensor [batch, seq_len, output_dim]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
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
