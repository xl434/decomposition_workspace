"""
Component: BN+ReLU+Conv+Dropout (in_features=64)
Source: data/kernelbench/level3/14_DenseNet121DenseBlock.py
Abstraction Level: fusion
Parent: dense_layer_1
Operations: [BatchNorm2d(64), ReLU, Conv2d(64, 32, k=3, p=1, bias=False), Dropout(0.0)]
Input Shapes: [(10, 64, 224, 224)]
Output Shapes: [(10, 32, 224, 224)]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.drop(x)
        return x


def get_inputs():
    return [torch.randn(10, 64, 224, 224)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 32, 224, 224)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
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
