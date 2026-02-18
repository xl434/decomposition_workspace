"""
Component: Linear(96, 384) - Mlp fc1
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: kernel
Parent: swin_mlp_block_ffn
Operations: [Linear(96, 384)]
Input Shapes: [10, 3136, 96]
Output Shapes: [10, 3136, 384]
Description: First linear layer in Mlp module, dim=96, hidden=96*4=384
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(96, 384)

    def forward(self, x):
        return self.fc1(x)


def get_inputs():
    return [torch.randn(10, 3136, 96)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 3136, 384)]


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
