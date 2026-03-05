import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Single Conv2d kernel: Conv2d(64, 64, kernel_size=3, padding=1).

    Input:  [batch_size, 64, H, W]  (used at multiple spatial sizes: 4x4, 2x2)
    Output: [batch_size, 64, H, W]
    """

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 64, 4, 4)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 64, 4, 4)


def run_tests():
    print("Testing Level 0 Kernel: Conv2d(64, 64, 3, p=1)")
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    # Also test with 2x2 spatial (used in block 5)
    inputs_2x2 = [torch.randn(2, 64, 2, 2)]
    with torch.no_grad():
        output_2x2 = model(*inputs_2x2)
    assert output_2x2.shape == (2, 64, 2, 2), f"Shape mismatch for 2x2: {output_2x2.shape}"
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     {expected}")
    print(f"  Also tested with 2x2 spatial: {output_2x2.shape}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    run_tests()
