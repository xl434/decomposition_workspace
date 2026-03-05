import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Full VGG16 model (test-sized variant with reduced channels and spatial dims).

    Architecture:
        Block 1: Conv(3,8,3,p=1)->ReLU->Conv(8,8,3,p=1)->ReLU->MaxPool(2,2)
        Block 2: Conv(8,16,3,p=1)->ReLU->Conv(16,16,3,p=1)->ReLU->MaxPool(2,2)
        Block 3: Conv(16,32,3,p=1)->ReLU->Conv(32,32,3,p=1)->ReLU->Conv(32,32,3,p=1)->ReLU->MaxPool(2,2)
        Block 4: Conv(32,64,3,p=1)->ReLU->Conv(64,64,3,p=1)->ReLU->Conv(64,64,3,p=1)->ReLU->MaxPool(2,2)
        Block 5: Conv(64,64,3,p=1)->ReLU->Conv(64,64,3,p=1)->ReLU->Conv(64,64,3,p=1)->ReLU->MaxPool(2,2)
        Flatten
        Classifier: Linear(64,256)->ReLU->Dropout->Linear(256,256)->ReLU->Dropout->Linear(256,10)

    Input: [batch_size, 3, 32, 32]
    Output: [batch_size, num_classes]
    """

    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_inputs():
    """Return a list of input tensors for the model."""
    batch_size = 2
    return [torch.randn(batch_size, 3, 32, 32)]


def get_init_inputs():
    """Return a list of arguments for model initialization."""
    num_classes = 10
    return [num_classes]


def get_expected_output_shape():
    """Return the expected output shape."""
    batch_size = 2
    num_classes = 10
    return (batch_size, num_classes)


def run_tests():
    """Run basic tests to verify the model."""
    print("Testing Level 3 Model: VGG16")

    # Initialize
    init_inputs = get_init_inputs()
    model = Model(*init_inputs)
    model.eval()

    # Forward pass
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected:     {expected_shape}")
    print("  PASSED")
    return True


if __name__ == "__main__":
    run_tests()
