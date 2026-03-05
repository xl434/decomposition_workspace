import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ReLU activation kernel (works for any input shape).

    Operation: max(0, x) element-wise.
    Input:  any shape
    Output: same shape as input
    """

    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 8, 32, 32)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 8, 32, 32)


def run_tests():
    print("Testing Level 0 Kernel: ReLU")
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    assert (output >= 0).all(), "ReLU output contains negative values"
    # Test with various shapes
    for shape in [(2, 16, 16, 16), (2, 32, 8, 8), (2, 64, 4, 4), (2, 256)]:
        test_input = torch.randn(*shape)
        with torch.no_grad():
            test_output = model(test_input)
        assert test_output.shape == shape, f"Shape mismatch for {shape}: {test_output.shape}"
        assert (test_output >= 0).all(), f"Negative values for shape {shape}"
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     {expected}")
    print("  Tested multiple shapes: all passed")
    print("  PASSED")
    return True


if __name__ == "__main__":
    run_tests()
