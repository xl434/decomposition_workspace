import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Block 2, Conv+ReLU pair 2: Conv2d(16,16,3,p=1) + ReLU.

    Input:  [batch_size, 16, 16, 16]
    Output: [batch_size, 16, 16, 16]
    """

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 16, 16, 16)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 16, 16, 16)


def run_tests():
    print("Testing Level 1 Fusion: Conv+ReLU 2_2")
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    assert (output >= 0).all(), "ReLU output contains negative values"
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     {expected}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    run_tests()
