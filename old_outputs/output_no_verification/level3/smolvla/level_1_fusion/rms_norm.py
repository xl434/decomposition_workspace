"""
Component: RMS Normalization
Abstraction Level: fusion
Parent: vlm_expert_transformer (layer)

Operations: float32 cast, square, mean, rsqrt, multiply by weight, cast back

Input Shapes:
  - hidden_states: [B, S, D] float32 (D can be 960 for VLM or 720 for expert)

Output Shapes:
  - output: [B, S, D] float32

Weight Shapes:
  - weight: [D] float32
"""
import torch
import torch.nn as nn

TEXT_RMS_NORM_EPS = 1e-5


class Model(nn.Module):
    """RMS Normalization."""
    def __init__(self, hidden_size=960, eps=TEXT_RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
      
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def get_inputs():
    return [torch.randn(1, 113, 960)]

def get_init_inputs():
    return [960]

def get_expected_output_shape():
    return [(1, 113, 960)]

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
