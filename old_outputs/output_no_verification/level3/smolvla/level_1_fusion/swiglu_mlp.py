"""
Component: SwiGLU MLP
Abstraction Level: fusion
Parent: vlm_expert_transformer (layer)

Operations: gate_proj Linear, SiLU activation, up_proj Linear, element-wise multiply,
            down_proj Linear

Input Shapes:
  - x: [B, S, hidden_size] float32

Output Shapes:
  - output: [B, S, hidden_size] float32

Weight Shapes:
  - gate_proj: Linear(hidden_size, intermediate_size, bias=False)
  - up_proj: Linear(hidden_size, intermediate_size, bias=False)
  - down_proj: Linear(intermediate_size, hidden_size, bias=False)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_HIDDEN_SIZE = 960
TEXT_INTERMEDIATE_SIZE = 2560


class Model(nn.Module):
    """SwiGLU MLP (used in Llama-style transformers)."""
    def __init__(self, hidden_size=TEXT_HIDDEN_SIZE, intermediate_size=TEXT_INTERMEDIATE_SIZE):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def get_inputs():
    return [torch.randn(1, 113, TEXT_HIDDEN_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 113, TEXT_HIDDEN_SIZE)]

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
