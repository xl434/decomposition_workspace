"""
Component: Grouped Query Attention (GQA)
Abstraction Level: fusion
Parent: vlm_expert_transformer (layer)

Operations: KV head expansion (repeat), transpose, MatMul(Q,K^T), scale,
            masked fill, softmax, MatMul(attn,V), reshape

Input Shapes:
  - attention_mask: [B, S_q, S_k] bool
  - query_states: [B, S_q, num_heads, head_dim] float32
  - key_states: [B, S_k, num_kv_heads, head_dim] float32
  - value_states: [B, S_k, num_kv_heads, head_dim] float32

Output Shapes:
  - att_output: [B, S_q, num_heads * head_dim] float32

Weight Shapes: (none - no learnable parameters)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_NUM_HEADS = 15
TEXT_NUM_KV_HEADS = 5
TEXT_HEAD_DIM = 64


class Model(nn.Module):
    """Grouped Query Attention computation (no projections, just attention)."""
    def __init__(self, num_attention_heads=TEXT_NUM_HEADS, num_key_value_heads=TEXT_NUM_KV_HEADS,
                 head_dim=TEXT_HEAD_DIM):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

    def forward(self, attention_mask, query_states, key_states, value_states):
        batch_size = query_states.shape[0]
        sequence_length = key_states.shape[1]

        # Expand KV heads to match Q heads
        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, self.num_key_value_heads, self.num_key_value_groups, self.head_dim
        ).reshape(batch_size, sequence_length, self.num_attention_heads, self.head_dim)

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, self.num_key_value_heads, self.num_key_value_groups, self.head_dim
        ).reshape(batch_size, sequence_length, self.num_attention_heads, self.head_dim)

        # Attention computation
        query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
        key_states = key_states.to(dtype=torch.float32).transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= self.head_dim ** -0.5
        att_weights = att_weights.to(dtype=torch.float32)

        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = F.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, self.num_attention_heads * self.head_dim)
        return att_output


def get_inputs():
    B, S = 1, 163
    mask = torch.ones(B, S, S, dtype=torch.bool)
    q = torch.randn(B, S, TEXT_NUM_HEADS, TEXT_HEAD_DIM)
    k = torch.randn(B, S, TEXT_NUM_KV_HEADS, TEXT_HEAD_DIM)
    v = torch.randn(B, S, TEXT_NUM_KV_HEADS, TEXT_HEAD_DIM)
    return [mask, q, k, v]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 163, TEXT_NUM_HEADS * TEXT_HEAD_DIM)]

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
