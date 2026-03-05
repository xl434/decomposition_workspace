"""
Step 6 Refactored: vision_sdpa decomposed into kernels.
Children: q_proj, k_proj, v_proj (linear), qk_matmul, scale+softmax, av_matmul (matmul), out_proj (linear)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)
import linear as linear_mod
import matmul as matmul_mod
import softmax as softmax_mod

VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_NUM_PATCHES = 1024

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = linear_mod.Model(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)
        self.k_proj = linear_mod.Model(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)
        self.v_proj = linear_mod.Model(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)
        self.out_proj = linear_mod.Model(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)
        self.qk_matmul = matmul_mod.Model()
        self.av_matmul = matmul_mod.Model()
        self.softmax = softmax_mod.Model(dim=-1)
        self.num_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
        self.scale = self.head_dim ** -0.5

    def forward(self, hidden_states):
        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = self.qk_matmul(queries, keys.transpose(2, 3)) * self.scale
        attn_weights = self.softmax(attn_weights)
        attn_output = self.av_matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output

def get_inputs():
    return [torch.randn(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]
def get_init_inputs():
    return []
def get_expected_output_shape():
    return [(1, VISION_NUM_PATCHES, VISION_HIDDEN_SIZE)]

if __name__ == "__main__":
    model = RefactoredModel(); model.eval()
    with torch.no_grad():
        out = model(*get_inputs())
        assert tuple(out.shape) == tuple(get_expected_output_shape()[0])
        print("PASS")
