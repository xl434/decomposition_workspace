import torch
import torch.nn as nn


class Model(nn.Module):
    """
    VGG16 Convolutional Block 1.

    Architecture: Conv2d(3,8,3,p=1)->ReLU->Conv2d(8,8,3,p=1)->ReLU->MaxPool2d(2,2)
    Input:  [batch_size, 3, 32, 32]
    Output: [batch_size, 8, 16, 16]
    """

    def __init__(self):
        super(Model, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 3, 32, 32)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 8, 16, 16)


def run_tests():
    print("Testing Level 2 Layer: Conv Block 1")
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
