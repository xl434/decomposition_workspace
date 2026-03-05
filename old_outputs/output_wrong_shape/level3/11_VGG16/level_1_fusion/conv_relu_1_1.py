import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Block 1, Conv+ReLU pair 1: Conv2d(3,8,3,p=1) + ReLU.

    Input:  [batch_size, 3, 32, 32]
    Output: [batch_size, 8, 32, 32]
    """

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 3, 32, 32)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 8, 32, 32)


def run_tests():
    print("Testing Level 1 Fusion: Conv+ReLU 1_1")
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    # Verify ReLU: all outputs >= 0
    assert (output >= 0).all(), "ReLU output contains negative values"
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     {expected}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    run_tests()
