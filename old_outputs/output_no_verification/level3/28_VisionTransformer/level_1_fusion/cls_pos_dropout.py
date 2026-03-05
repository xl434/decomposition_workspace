"""
Component: cls_pos_dropout
Source: data/kernelbench/level3/28_VisionTransformer.py
Abstraction Level: fusion
Parent: patch_embedding
Operations: [cls_token expand+cat, pos_embedding add, dropout]
Input Shapes: [2, 196, 512]
Output Shapes: [2, 197, 512]
Children: [cls_token_expand, pos_embedding_add, dropout_0]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_patches=196, dim=512, emb_dropout=0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        # Expand cls_token and concatenate
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embedding
        x = x + self.pos_embedding
        # Apply dropout
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.randn(2, 196, 512)]


def get_init_inputs():
    return [196, 512, 0.0]


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
