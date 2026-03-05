import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Linear kernel: Linear(256, 10).

    Input:  [batch_size, 256]
    Output: [batch_size, 10]
    """

    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        return self.linear(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 256)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 10)


def run_tests():
    print("Testing Level 0 Kernel: Linear(256, 10)")
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
