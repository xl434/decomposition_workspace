import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Single Conv2d kernel: Conv2d(8, 16, kernel_size=3, padding=1).

    Input:  [batch_size, 8, 16, 16]
    Output: [batch_size, 16, 16, 16]
    """

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(8, 16, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 8, 16, 16)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 16, 16, 16)


def run_tests():
    print("Testing Level 0 Kernel: Conv2d(8, 16, 3, p=1)")
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     {expected}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    run_tests()
