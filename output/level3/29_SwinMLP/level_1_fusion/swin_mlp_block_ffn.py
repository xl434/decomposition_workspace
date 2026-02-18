"""
Component: SwinMLPBlock FFN Branch
Source: data/kernelbench/level3/29_SwinMLP.py
Abstraction Level: fusion
Parent: basic_layer_0
Operations: [LayerNorm, Linear(96,384), GELU, Dropout, Linear(384,96), Dropout, residual_add]
Input Shapes: [10, 3136, 96]
Output Shapes: [10, 3136, 96]
Description: The FFN (feed-forward network) branch of SwinMLPBlock.
  Applies norm2, then Mlp(96->384->96) with GELU, then adds residual.
  Stage 0 params: dim=96, mlp_ratio=4.0
"""
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 96
        mlp_ratio = 4.0
        mlp_hidden_dim = int(dim * mlp_ratio)  # 384
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_inputs():
    return [torch.randn(10, 3136, 96)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(10, 3136, 96)]


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
