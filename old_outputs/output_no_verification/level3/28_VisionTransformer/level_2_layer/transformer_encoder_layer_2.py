"""
Component: transformer_encoder_layer_2
Source: data/kernelbench/level3/28_VisionTransformer.py
Abstraction Level: layer
Parent: vision_transformer
Operations: [TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.0)]
Input Shapes: [2, 197, 512]
Output Shapes: [2, 197, 512]
Children: [self_attention, ffn_block]
Note: Layer 2 of the 6-layer TransformerEncoder stack
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x):
        # Original model passes [batch, seq, dim] directly without transposing
        x = self.encoder_layer(x)
        return x


def get_inputs():
    return [torch.randn(2, 197, 512)]


def get_init_inputs():
    return [512, 8, 2048, 0.0]


def get_expected_output_shape():
    return [(2, 197, 512)]


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
