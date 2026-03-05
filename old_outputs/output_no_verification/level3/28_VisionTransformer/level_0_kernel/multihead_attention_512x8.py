"""
Component: multihead_attention_512x8
Source: data/kernelbench/level3/28_VisionTransformer.py
Abstraction Level: kernel
Parent: self_attention
Operations: [nn.MultiheadAttention(embed_dim=512, num_heads=8)]
Input Shapes: [2, 197, 512] - matching original model's data flow (no transpose)
Output Shapes: [2, 197, 512]
Note: Original model passes [batch, seq, dim] directly to TransformerEncoder without transposing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # Self-attention: query=key=value=x
        output, _ = self.mha(x, x, x)
        return output


def get_inputs():
    # Original model passes [batch, seq, dim] without transposing
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
