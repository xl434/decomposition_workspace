"""
Component: mlp_head_fusion
Source: data/kernelbench/level3/28_VisionTransformer.py
Abstraction Level: fusion
Parent: mlp_head
Operations: [Linear(512,2048) + GELU + Dropout(0.0) + Linear(2048,10)]
Input Shapes: [2, 512]
Output Shapes: [2, 10]
Children: [linear_512x2048_head, gelu, linear_2048x10]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim=512, mlp_dim=2048, num_classes=10, dropout=0.0):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp_head(x)


def get_inputs():
    return [torch.randn(2, 512)]


def get_init_inputs():
    return [512, 2048, 10, 0.0]


def get_expected_output_shape():
    return [(2, 10)]


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
