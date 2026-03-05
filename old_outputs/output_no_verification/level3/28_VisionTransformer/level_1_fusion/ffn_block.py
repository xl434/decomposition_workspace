"""
Component: ffn_block
Source: data/kernelbench/level3/28_VisionTransformer.py
Abstraction Level: fusion
Parent: transformer_encoder_layer_*
Operations: [Linear(512,2048) + ReLU + Dropout + Linear(2048,512) + residual + LayerNorm(512)]
Input Shapes: [2, 197, 512]
Output Shapes: [2, 197, 512]
Children: [linear_512x2048, relu_2048, dropout_ffn, linear_2048x512, layer_norm_512_2]
Note: FFN sub-block of TransformerEncoderLayer. PyTorch default uses ReLU and post-LN.
      x = norm2(x + dropout(linear2(dropout(relu(linear1(x))))))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, d_model=512, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # FFN: Linear -> ReLU -> Dropout -> Linear -> Dropout
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        # Residual + LayerNorm
        x = self.norm2(x + self.dropout2(ff))
        return x


def get_inputs():
    return [torch.randn(2, 197, 512)]


def get_init_inputs():
    return [512, 2048, 0.0]


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
