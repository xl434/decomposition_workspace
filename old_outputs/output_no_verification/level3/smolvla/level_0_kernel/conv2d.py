"""
Component: 2D Convolution
Abstraction Level: kernel
Operations: conv2d

Input Shapes:
  - x: [B, C_in, H, W] float32

Output Shapes:
  - out: [B, C_out, H_out, W_out] float32

Weight Shapes:
  - weight: [C_out, C_in, kH, kW]
  - bias: [C_out]
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """2D Convolution kernel."""
    def __init__(self, in_channels=3, out_channels=768, kernel_size=16, stride=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.randn(1, 3, 512, 512)]

def get_init_inputs():
    return [3, 768, 16, 16]

def get_expected_output_shape():
    return [(1, 768, 32, 32)]

def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected = get_expected_output_shape()
            assert tuple(output.shape) == tuple(expected[0]), \
                f"Shape mismatch: {output.shape} vs {expected[0]}"
            print(f"Input shape: {inputs[0].shape}")
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
