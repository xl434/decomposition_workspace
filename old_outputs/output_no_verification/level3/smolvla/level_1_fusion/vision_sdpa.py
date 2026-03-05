"""
Component: Vision Scaled Dot-Product Attention
Abstraction Level: fusion
Parent: vision_encoder (layer)

Operations: Q/K/V Linear projections, reshape to heads, MatMul(Q,K^T), scale,
            softmax, MatMul(attn,V), reshape, output Linear projection

Input Shapes:
  - hidden_states: [B, 1024, 768] float32

Output Shapes:
  - attn_output: [B, 1024, 768] float32

Weight Shapes:
  - q_proj: Linear(768, 768)
  - k_proj: Linear(768, 768)
  - v_proj: Linear(768, 768)
  - out_proj: Linear(768, 768)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_HEAD_DIM = 64
VISION_NUM_PATCHES = 1024


class Model(nn.Module):
    """Vision multi-head self-attention."""
    def __init__(self):
        super().__init__()
        self.embed_dim = VISION_HIDDEN_SIZE
        self.num_heads = VISION_NUM_HEADS
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(2, 3)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


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
