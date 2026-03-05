"""
Composition test for 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

Verifies that chaining the individual kernel components produces
identical output to the original fused model, with shared weights.
"""

import torch
import torch.nn as nn


def test_composition():
    torch.manual_seed(42)

    # Original model (fused)
    class OriginalModel(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
            super().__init__()
            self.conv_transpose = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
            self.batch_norm = nn.BatchNorm3d(out_channels)
            self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
            self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

        def forward(self, x):
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = self.avg_pool1(x)
            x = self.avg_pool2(x)
            return x

    # Composed model (chaining individual kernels)
    class ComposedModel(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
            super().__init__()
            # Step 1: conv_transpose3d kernel
            self.conv_transpose = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
            # Step 2: batch_norm3d kernel
            self.batch_norm = nn.BatchNorm3d(out_channels)
            # Step 3: avg_pool3d_1 kernel
            self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
            # Step 4: avg_pool3d_2 kernel
            self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

        def forward(self, x):
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = self.avg_pool1(x)
            x = self.avg_pool2(x)
            return x

    in_channels = 2
    out_channels = 4
    kernel_size = 3
    stride = 2
    padding = 1
    bias_shape = (4, 1, 1, 1)

    original = OriginalModel(in_channels, out_channels, kernel_size, stride, padding, bias_shape)
    composed = ComposedModel(in_channels, out_channels, kernel_size, stride, padding)

    # Copy weights from original to composed
    # ConvTranspose3d weights
    composed.conv_transpose.weight = original.conv_transpose.weight
    composed.conv_transpose.bias = original.conv_transpose.bias

    # BatchNorm3d weights and buffers
    composed.batch_norm.weight = original.batch_norm.weight
    composed.batch_norm.bias = original.batch_norm.bias
    composed.batch_norm.running_mean.copy_(original.batch_norm.running_mean)
    composed.batch_norm.running_var.copy_(original.batch_norm.running_var)
    composed.batch_norm.num_batches_tracked.copy_(original.batch_norm.num_batches_tracked)

    original.eval()
    composed.eval()

    with torch.no_grad():
        x = torch.randn(2, 2, 8, 8, 8)
        out_orig = original(x)
        out_comp = composed(x)

        max_diff = (out_orig - out_comp).abs().max().item()
        print(f"Original output shape: {out_orig.shape}")
        print(f"Composed output shape: {out_comp.shape}")
        print(f"Max difference: {max_diff}")

        assert out_orig.shape == out_comp.shape, (
            f"Shape mismatch: original={out_orig.shape}, composed={out_comp.shape}"
        )
        assert torch.allclose(out_orig, out_comp, rtol=1e-4, atol=1e-5), (
            f"FAILED: max_diff={max_diff}"
        )
        print("PASS - Composition test passed")


if __name__ == "__main__":
    test_composition()
