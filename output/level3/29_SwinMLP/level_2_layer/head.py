"""
Component: Classification Head (Norm + AvgPool + Flatten + Linear)
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: layer
Parent: swin_mlp
Operations: [LayerNorm(768), transpose(1,2), AdaptiveAvgPool1d(1), flatten(1), Linear(768,1000)]
Input Shapes: [10, 49, 768]
Output Shapes: [10, 1000]
Description: Final classification head. Applies LayerNorm, global average pooling,
  flattening, and linear classification layer.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(768, 1000)

    def forward(self, x):
        x = self.norm(x)                      # [10, 49, 768]
        x = self.avgpool(x.transpose(1, 2))   # [10, 768, 1]
        x = torch.flatten(x, 1)               # [10, 768]
        x = self.head(x)                       # [10, 1000]
        return x


def get_inputs():
    return [torch.randn(10, 49, 768)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 1000)]


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
