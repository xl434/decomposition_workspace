"""
Component: DenseNet121 DenseBlock
Source: data/kernelbench/level3/14_DenseNet121DenseBlock.py
Abstraction Level: model
Parent: None
Operations: [DenseLayer0, DenseLayer1, DenseLayer2, DenseLayer3, DenseLayer4, DenseLayer5]
Input Shapes: [(10, 32, 224, 224)]
Output Shapes: [(10, 224, 224, 224)]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x


def get_inputs():
    return [torch.randn(10, 32, 224, 224)]


def get_init_inputs():
    return [6, 32, 32]


def get_expected_output_shape():
    return [(10, 224, 224, 224)]


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
