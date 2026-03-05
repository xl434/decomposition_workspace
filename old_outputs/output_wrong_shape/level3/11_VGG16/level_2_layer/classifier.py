import torch
import torch.nn as nn


class Model(nn.Module):
    """
    VGG16 Classifier head.

    Architecture: Linear(64,256)->ReLU->Dropout(0.0)->Linear(256,256)->ReLU->Dropout(0.0)->Linear(256,10)
    Input:  [batch_size, 64]
    Output: [batch_size, 10]
    """

    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, 64)]


def get_init_inputs():
    num_classes = 10
    return [num_classes]


def get_expected_output_shape():
    return (2, 10)


def run_tests():
    print("Testing Level 2 Layer: Classifier")
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
