import torch
import torch.nn as nn


class Model(nn.Module):
    """
    MaxPool2d kernel: MaxPool2d(kernel_size=2, stride=2).

    Halves spatial dimensions.
    Input:  [batch_size, C, H, W]
    Output: [batch_size, C, H//2, W//2]
    """

    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 8, 32, 32)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 8, 16, 16)


def run_tests():
    print("Testing Level 0 Kernel: MaxPool2d(2, 2)")
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    # Test with various shapes
    test_cases = [
        ((2, 8, 32, 32), (2, 8, 16, 16)),
        ((2, 16, 16, 16), (2, 16, 8, 8)),
        ((2, 32, 8, 8), (2, 32, 4, 4)),
        ((2, 64, 4, 4), (2, 64, 2, 2)),
        ((2, 64, 2, 2), (2, 64, 1, 1)),
    ]
    for in_shape, out_shape in test_cases:
        test_input = torch.randn(*in_shape)
        with torch.no_grad():
            test_output = model(test_input)
        assert test_output.shape == out_shape, (
            f"Shape mismatch for {in_shape}: got {test_output.shape}, expected {out_shape}"
        )
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     {expected}")
    print("  Tested all VGG16 pool sizes: all passed")
    print("  PASSED")
    return True


if __name__ == "__main__":
    run_tests()
