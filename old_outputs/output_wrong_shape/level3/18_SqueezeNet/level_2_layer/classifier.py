"""
Component: classifier
Abstraction Level: layer
Parent: squeezenet

Operations: [Dropout, Conv2d, ReLU, AdaptiveAvgPool2d]

Input Shapes:
  - x: (batch_size, 512, height, width) dtype=float32

Output Shapes:
  - output: (batch_size, num_classes, 1, 1) dtype=float32

Weight Shapes:
  - conv.weight: (num_classes, 512, 1, 1)
  - conv.bias: (num_classes,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Classifier: Dropout(p=0.0) -> Conv2d(512,num_classes,1) -> ReLU -> AdaptiveAvgPool2d(1,1)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.classifier(x)

def get_inputs():
    """Generate test inputs."""
    batch_size = 2
    return [torch.randn(batch_size, 512, 1, 1, dtype=torch.float32)]

def get_init_inputs():
    """Return initialization parameters."""
    return [10]  # num_classes

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    batch_size = 2
    num_classes = 10
    return [(batch_size, num_classes, 1, 1)]

def run_tests():
    """Verify this component executes correctly."""
    try:
        model = Model(*get_init_inputs())
        model.eval()

        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)

            # 1. Basic validation
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            # 2. Shape validation
            expected_shapes = get_expected_output_shape()
            actual_shape = output.shape

            assert tuple(actual_shape) == tuple(expected_shapes[0]), \
                f"Shape mismatch: got {actual_shape}, expected {expected_shapes[0]}"

            # 3. Dtype validation
            expected_dtype = inputs[0].dtype
            assert output.dtype == expected_dtype, f"Dtype mismatch: {output.dtype} vs {expected_dtype}"

            print(f"Input shape: {inputs[0].shape}")
            print(f"Output shape: {actual_shape}")
            print("PASS")
            return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
