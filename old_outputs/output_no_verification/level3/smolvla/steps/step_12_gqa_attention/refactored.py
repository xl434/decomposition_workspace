"""
Step 12 Refactored: gqa_attention decomposed into kernels.
Children: qk_matmul (matmul), softmax (softmax), av_matmul (matmul)
forward() does: KV head expansion (reshape), dtype casts, masking, reshaping
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)
import matmul as matmul_mod
import softmax as softmax_mod

TEXT_NUM_HEADS = 15
TEXT_NUM_KV_HEADS = 5
TEXT_HEAD_DIM = 64

class RefactoredModel(nn.Module):
    def __init__(self, num_attention_heads=TEXT_NUM_HEADS, num_key_value_heads=TEXT_NUM_KV_HEADS,
                 head_dim=TEXT_HEAD_DIM):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.qk_matmul = matmul_mod.Model()
        self.softmax = softmax_mod.Model(dim=-1)
        self.av_matmul = matmul_mod.Model()

    def forward(self, attention_mask, query_states, key_states, value_states):
        batch_size = query_states.shape[0]
        sequence_length = key_states.shape[1]

        # Expand KV heads (reshape/expand - allowed data plumbing)
        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, self.num_key_value_heads, self.num_key_value_groups, self.head_dim
        ).reshape(batch_size, sequence_length, self.num_attention_heads, self.head_dim)

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, self.num_key_value_heads, self.num_key_value_groups, self.head_dim
        ).reshape(batch_size, sequence_length, self.num_attention_heads, self.head_dim)

        # Transpose for attention
        query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
        key_states = key_states.to(dtype=torch.float32).transpose(1, 2)

        # Q @ K^T
        att_weights = self.qk_matmul(query_states, key_states.transpose(2, 3))
        att_weights = att_weights * (self.head_dim ** -0.5)
        att_weights = att_weights.to(dtype=torch.float32)

        # Masking (allowed - uses tensor indexing and fill value)
        big_neg = torch.finfo(att_weights.dtype).min
        att_weights = attention_mask[:, None, :, :].to(torch.float32) * att_weights + \
                      (~attention_mask[:, None, :, :]).to(torch.float32) * big_neg

        # Softmax
        probs = self.softmax(att_weights)
        probs = probs.to(dtype=value_states.dtype)

        # Attn @ V
        att_output = self.av_matmul(probs, value_states.permute(0, 2, 1, 3))
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

if __name__ == "__main__":
    model = RefactoredModel(); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
