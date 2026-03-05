"""
Component: patch_embedding
Source: data/kernelbench/level3/28_VisionTransformer.py
Abstraction Level: layer
Parent: vision_transformer
Operations: [unfold+reshape, Linear(768,512), cls_token expand+cat, pos_embedding add, Dropout(0.0)]
Input Shapes: [2, 3, 224, 224]
Output Shapes: [2, 197, 512]
Children: [patch_extract_embed, cls_pos_dropout]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dim=512, channels=3, emb_dropout=0.0):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2  # 196
        patch_dim = channels * patch_size ** 2  # 768

        self.patch_size = patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, img):
        p = self.patch_size
        # Patch extraction
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p * p * img.shape[1])
        # Linear embedding
        x = self.patch_to_embedding(x)
        # CLS token
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Positional embedding
        x = x + self.pos_embedding
        # Dropout
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.randn(2, 3, 224, 224)]


def get_init_inputs():
    return [224, 16, 512, 3, 0.0]


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
