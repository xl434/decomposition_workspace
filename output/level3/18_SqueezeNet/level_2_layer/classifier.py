"""
Component: Classifier
Source: data/kernelbench/level3/18_SqueezeNet.py
Abstraction Level: layer
Parent: squeezenet.py
Operations: [Dropout(0.0), Conv2d(512,1000,k=1), ReLU, AdaptiveAvgPool2d(1,1), Flatten]
Input Shapes: [64, 512, 31, 31]
Output Shapes: [64, 1000]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(p=0.0)
        self.conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def get_inputs():
    return [torch.randn(64, 512, 31, 31)]


def get_init_inputs():
    return [1000]


def get_expected_output_shape():
    return [(64, 1000)]


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
