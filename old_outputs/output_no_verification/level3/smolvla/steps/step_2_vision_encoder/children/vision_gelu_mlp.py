"""
Component: Vision GELU MLP
Abstraction Level: fusion
Parent: vision_encoder (layer)

Operations: Linear, GELU(tanh approx), Linear

Input Shapes:
  - hidden_states: [B, 1024, 768] float32

Output Shapes:
  - output: [B, 1024, 768] float32

Weight Shapes:
  - fc1: Linear(768, 3072)
  - fc2: Linear(3072, 768)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VISION_HIDDEN_SIZE = 768
VISION_INTERMEDIATE_SIZE = 3072
VISION_NUM_PATCHES = 1024


class Model(nn.Module):
    """Vision MLP with GELU activation."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE)
        self.fc2 = nn.Linear(VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate='tanh')
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.randn(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]

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
