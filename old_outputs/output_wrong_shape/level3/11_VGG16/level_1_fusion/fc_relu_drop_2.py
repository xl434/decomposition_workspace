import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Classifier FC block 2: Linear(256,256) + ReLU + Dropout(0.0).

    Input:  [batch_size, 256]
    Output: [batch_size, 256]
    """

    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        return self.dropout(self.relu(self.linear(x)))


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 256)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return (2, 256)


def run_tests():
    print("Testing Level 1 Fusion: FC+ReLU+Dropout 2")
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
