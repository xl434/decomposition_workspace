import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Flatten kernel: torch.flatten(x, 1).

    Flattens all dimensions except the batch dimension.
    Input:  [batch_size, C, H, W]
    Output: [batch_size, C*H*W]
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 64, 1, 1)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 64)


def run_tests():
    print("Testing Level 0 Kernel: Flatten")
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
