"""
Component: Pixel Shuffle + Linear Projection (Connector)
Abstraction Level: fusion
Parent: vision_encoder (layer)

Operations: reshape (pixel shuffle), Linear projection

Input Shapes:
  - image_hidden_states: [B, 1024, 768] float32

Output Shapes:
  - output: [B, 64, 960] float32

Weight Shapes:
  - proj: Linear(12288, 960, bias=False)
"""
import torch
import torch.nn as nn

VISION_HIDDEN_SIZE = 768
SCALE_FACTOR = 4
CONNECTOR_INPUT_DIM = VISION_HIDDEN_SIZE * (SCALE_FACTOR ** 2)  # 12288
CONNECTOR_OUTPUT_DIM = 960
VISION_NUM_PATCHES = 1024


class Model(nn.Module):
    """Pixel shuffle connector: rearranges patches and projects to text hidden dim."""
    def __init__(self):
        super().__init__()
        self.scale_factor = SCALE_FACTOR
        self.proj = nn.Linear(CONNECTOR_INPUT_DIM, CONNECTOR_OUTPUT_DIM, bias=False)

    def forward(self, image_hidden_states):
        # Pixel shuffle
        bsz, seq, embed_dim = image_hidden_states.size()
        height = width = int(seq ** 0.5)
        sf = self.scale_factor
        x = image_hidden_states.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / sf), embed_dim * sf)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / sf), int(height / sf), embed_dim * (sf ** 2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (sf ** 2)), embed_dim * (sf ** 2))
        # Linear projection
        x = self.proj(x)
        return x


def get_inputs():
    return [torch.randn(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 64, CONNECTOR_OUTPUT_DIM)]

def run_tests():
    try:
        model = Model()
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected = get_expected_output_shape()
            assert tuple(output.shape) == tuple(expected[0]), \
                f"Shape mismatch: {output.shape} vs {expected[0]}"
            print(f"Input shape: {inputs[0].shape}")
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
