"""
Component: self_attention
Source: data/kernelbench/level3/28_VisionTransformer.py
Abstraction Level: fusion
Parent: transformer_encoder_layer_*
Operations: [MultiheadAttention(512, 8) + residual + LayerNorm(512)]
Input Shapes: [2, 197, 512]
Output Shapes: [2, 197, 512]
Children: [multihead_attention_512x8, layer_norm_512]
Note: Implements the self-attention sub-block of TransformerEncoderLayer (pre-norm is post-LN in PyTorch default)
      PyTorch default: x = norm1(x + self_attn(x, x, x))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        # PyTorch TransformerEncoderLayer default: post-LN
        # x = norm1(x + dropout(self_attn(x, x, x)))
        # Original model passes [batch, seq, dim] directly to TransformerEncoder
        # without transposing, so MHA receives data as-is
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        return x


def get_inputs():
    return [torch.randn(2, 197, 512)]


def get_init_inputs():
    return [512, 8, 0.0]


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
