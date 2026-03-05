import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Dropout kernel: Dropout(p=0.0).

    With p=0.0, this is effectively a no-op (identity function).
    Input:  any shape
    Output: same shape as input
    """

    def __init__(self):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        return self.dropout(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 256)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 256)


def run_tests():
    print("Testing Level 0 Kernel: Dropout(p=0.0)")
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected = get_expected_output_shape()
    assert output.shape == expected, f"Shape mismatch: {output.shape} vs {expected}"
    # With p=0.0 in eval mode, output should equal input exactly
    assert torch.allclose(output, inputs[0]), "Dropout(0.0) should be identity in eval mode"
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     {expected}")
    print("  Verified identity behavior with p=0.0")
    print("  PASSED")
    return True


if __name__ == "__main__":
    run_tests()
