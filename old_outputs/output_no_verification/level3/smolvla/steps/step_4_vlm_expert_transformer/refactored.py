"""
Step 4 Refactored: vlm_expert_transformer decomposed into fusion-level children.

Children:
  - rope: RoPE embedding (L1 fusion, no params)
  - gqa_attn: GQA attention (L1 fusion, no params)
  - vlm_input_ln: 16x RMSNorm(960) (L1 fusion)
  - vlm_post_ln: 16x RMSNorm(960) (L1 fusion)
  - vlm_mlp: 16x SwiGLU MLP(960,2560) (L1 fusion)
  - vlm_q_proj: 16x nn.Linear(960,960) (L0 kernel)
  - vlm_k_proj: 16x nn.Linear(960,320) (L0 kernel)
  - vlm_v_proj: 16x nn.Linear(960,320) (L0 kernel)
  - vlm_o_proj: 16x nn.Linear(960,960) (L0 kernel)
  - exp_input_ln: 16x RMSNorm(720) (L1 fusion)
  - exp_post_ln: 16x RMSNorm(720) (L1 fusion)
  - exp_mlp: 16x SwiGLU MLP(720,2048) (L1 fusion)
  - exp_q_proj: 16x nn.Linear(720,960) (L0 kernel)
  - exp_k_proj: 16x nn.Linear - self-attn(720,320) or cross-attn(320,320) (L0 kernel)
  - exp_v_proj: 16x nn.Linear - self-attn(720,320) or cross-attn(320,320) (L0 kernel)
  - exp_o_proj: 16x nn.Linear(960,720) (L0 kernel)
  - vlm_norm: RMSNorm(960) (L1 fusion)
  - exp_norm: RMSNorm(720) (L1 fusion)

forward() only does data plumbing: concat, reshape/view, slicing, residual add, looping.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)

import rms_norm as rms_mod
import swiglu_mlp as swiglu_mod
import gqa_attention as gqa_mod
import rope_embedding as rope_mod

TEXT_HIDDEN_SIZE = 960
TEXT_NUM_HEADS = 15
TEXT_NUM_KV_HEADS = 5
TEXT_HEAD_DIM = 64
TEXT_INTERMEDIATE_SIZE = 2560
TEXT_RMS_NORM_EPS = 1e-5
NUM_VLM_LAYERS = 16
EXPERT_HIDDEN_SIZE = 720
EXPERT_INTERMEDIATE_SIZE = 2048
EXPERT_NUM_HEADS = 15
EXPERT_NUM_KV_HEADS = 5
SELF_ATTN_EVERY_N_LAYERS = 2
CHUNK_SIZE = 50


class RefactoredModel(nn.Module):
    """VLM+Expert Transformer decomposed into fusion-level children."""

    def __init__(self):
        super().__init__()
        # Shared stateless children
        self.rope = rope_mod.Model()
        self.gqa_attn = gqa_mod.Model()

        # VLM per-layer children
        self.vlm_input_ln = nn.ModuleList([rms_mod.Model(TEXT_HIDDEN_SIZE) for _ in range(NUM_VLM_LAYERS)])
        self.vlm_post_ln = nn.ModuleList([rms_mod.Model(TEXT_HIDDEN_SIZE) for _ in range(NUM_VLM_LAYERS)])
        self.vlm_mlp = nn.ModuleList([swiglu_mod.Model(TEXT_HIDDEN_SIZE, TEXT_INTERMEDIATE_SIZE) for _ in range(NUM_VLM_LAYERS)])
        self.vlm_q_proj = nn.ModuleList([nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_HEADS * TEXT_HEAD_DIM, bias=False) for _ in range(NUM_VLM_LAYERS)])
        self.vlm_k_proj = nn.ModuleList([nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM, bias=False) for _ in range(NUM_VLM_LAYERS)])
        self.vlm_v_proj = nn.ModuleList([nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM, bias=False) for _ in range(NUM_VLM_LAYERS)])
        self.vlm_o_proj = nn.ModuleList([nn.Linear(TEXT_NUM_HEADS * TEXT_HEAD_DIM, TEXT_HIDDEN_SIZE, bias=False) for _ in range(NUM_VLM_LAYERS)])

        # Expert per-layer children
        self.exp_input_ln = nn.ModuleList([rms_mod.Model(EXPERT_HIDDEN_SIZE) for _ in range(NUM_VLM_LAYERS)])
        self.exp_post_ln = nn.ModuleList([rms_mod.Model(EXPERT_HIDDEN_SIZE) for _ in range(NUM_VLM_LAYERS)])
        self.exp_mlp = nn.ModuleList([swiglu_mod.Model(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE) for _ in range(NUM_VLM_LAYERS)])
        self.exp_q_proj = nn.ModuleList([nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_HEADS * TEXT_HEAD_DIM, bias=False) for _ in range(NUM_VLM_LAYERS)])
        self.exp_o_proj = nn.ModuleList([nn.Linear(EXPERT_NUM_HEADS * TEXT_HEAD_DIM, EXPERT_HIDDEN_SIZE, bias=False) for _ in range(NUM_VLM_LAYERS)])

        # Expert K/V projections: self-attn layers (even) take expert input, cross-attn layers (odd) take VLM KV
        exp_k_projs = []
        exp_v_projs = []
        vlm_kv_dim = TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM  # 320
        expert_kv_dim = EXPERT_NUM_KV_HEADS * TEXT_HEAD_DIM  # 320
        for layer_idx in range(NUM_VLM_LAYERS):
            if SELF_ATTN_EVERY_N_LAYERS > 0 and layer_idx % SELF_ATTN_EVERY_N_LAYERS == 0:
                # Self-attention: KV from expert hidden
                exp_k_projs.append(nn.Linear(EXPERT_HIDDEN_SIZE, expert_kv_dim, bias=False))
                exp_v_projs.append(nn.Linear(EXPERT_HIDDEN_SIZE, expert_kv_dim, bias=False))
            else:
                # Cross-attention: KV from VLM (already projected to vlm_kv_dim)
                exp_k_projs.append(nn.Linear(vlm_kv_dim, expert_kv_dim, bias=False))
                exp_v_projs.append(nn.Linear(vlm_kv_dim, expert_kv_dim, bias=False))
        self.exp_k_proj = nn.ModuleList(exp_k_projs)
        self.exp_v_proj = nn.ModuleList(exp_v_projs)

        # Final norms
        self.vlm_norm = rms_mod.Model(TEXT_HIDDEN_SIZE)
        self.exp_norm = rms_mod.Model(EXPERT_HIDDEN_SIZE)

    def forward(self, prefix_embs, suffix_embs, att_2d_masks, position_ids):
        inputs_embeds = [prefix_embs, suffix_embs]
        batch_size = prefix_embs.shape[0]

        for layer_idx in range(NUM_VLM_LAYERS):
            is_self_attn = (SELF_ATTN_EVERY_N_LAYERS > 0 and layer_idx % SELF_ATTN_EVERY_N_LAYERS == 0)

            if is_self_attn:
                # Self-attention: all streams participate in joint attention
                # VLM QKV
                vlm_hn = self.vlm_input_ln[layer_idx](inputs_embeds[0])
                vlm_shape = (*vlm_hn.shape[:-1], -1, TEXT_HEAD_DIM)
                vlm_q = self.vlm_q_proj[layer_idx](vlm_hn).view(vlm_shape)
                vlm_k = self.vlm_k_proj[layer_idx](vlm_hn).view(vlm_shape)
                vlm_v = self.vlm_v_proj[layer_idx](vlm_hn).view(vlm_shape)

                # Expert QKV
                exp_hn = self.exp_input_ln[layer_idx](inputs_embeds[1])
                exp_shape = (*exp_hn.shape[:-1], -1, TEXT_HEAD_DIM)
                exp_q = self.exp_q_proj[layer_idx](exp_hn).view(exp_shape)
                exp_k = self.exp_k_proj[layer_idx](exp_hn).view(exp_shape)
                exp_v = self.exp_v_proj[layer_idx](exp_hn).view(exp_shape)

                # Concatenate all streams
                all_q = torch.cat([vlm_q, exp_q], dim=1)
                all_k = torch.cat([vlm_k, exp_k], dim=1)
                all_v = torch.cat([vlm_v, exp_v], dim=1)

                seq_len = all_q.shape[1]
                if seq_len < position_ids.shape[1]:
                    _pos = position_ids[:, :seq_len]
                    _mask = att_2d_masks[:, :seq_len, :seq_len]
                else:
                    _pos = position_ids
                    _mask = att_2d_masks

                # Apply RoPE
                all_q = self.rope(all_q, _pos)
                all_k = self.rope(all_k, _pos)

                # GQA attention
                att_output = self.gqa_attn(_mask, all_q, all_k, all_v)
                att_outputs = [att_output]
            else:
                # Cross-attention: VLM does self-attn, expert cross-attends to VLM KV
                p_len = inputs_embeds[0].shape[1]
                pos_id = position_ids[:, :p_len]
                exp_pos_id = position_ids[:, p_len:]
                prefix_mask = att_2d_masks[:, :p_len, :p_len]

                # VLM self-attention
                vlm_hn = self.vlm_input_ln[layer_idx](inputs_embeds[0])
                vlm_shape = (*vlm_hn.shape[:-1], -1, TEXT_HEAD_DIM)
                vlm_q = self.vlm_q_proj[layer_idx](vlm_hn).view(vlm_shape)
                key_state = self.vlm_k_proj[layer_idx](vlm_hn).view(vlm_shape)
                value_states = self.vlm_v_proj[layer_idx](vlm_hn).view(vlm_shape)

                vlm_q = self.rope(vlm_q, pos_id)
                key_states = self.rope(key_state, pos_id)

                vlm_att = self.gqa_attn(prefix_mask, vlm_q, key_states, value_states)
                att_outputs = [vlm_att]

                # Expert cross-attention
                exp_hn = self.exp_input_ln[layer_idx](inputs_embeds[1])
                exp_shape = (*exp_hn.shape[:-1], -1, TEXT_HEAD_DIM)
                exp_q = self.exp_q_proj[layer_idx](exp_hn).view(exp_shape)

                # Reproject VLM KV for expert
                _ks = key_states.reshape(*key_states.shape[:2], -1)
                exp_k = self.exp_k_proj[layer_idx](_ks).view(*_ks.shape[:-1], -1, TEXT_HEAD_DIM)
                _vs = value_states.reshape(*value_states.shape[:2], -1)
                exp_v = self.exp_v_proj[layer_idx](_vs).view(*_vs.shape[:-1], -1, TEXT_HEAD_DIM)

                exp_pos_id = exp_pos_id - torch.min(exp_pos_id, dim=1, keepdim=True).values
                exp_mask = att_2d_masks[:, -inputs_embeds[1].shape[1]:, :exp_k.shape[1]:]

                exp_q = self.rope(exp_q, exp_pos_id)
                exp_att = self.gqa_attn(exp_mask, exp_q, exp_k, exp_v)
                att_outputs.append(exp_att)

            # Post-attention: o_proj + residual + MLP
            outputs_embeds = []
            start = 0
            layer_configs = [
                (0, self.vlm_o_proj[layer_idx], self.vlm_post_ln[layer_idx], self.vlm_mlp[layer_idx]),
                (1, self.exp_o_proj[layer_idx], self.exp_post_ln[layer_idx], self.exp_mlp[layer_idx]),
            ]
            for i, o_proj, post_ln, mlp in layer_configs:
                hs = inputs_embeds[i]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                end = start + hs.shape[1]
                att_slice = att_output[:, start:end]
                if att_slice.dtype != o_proj.weight.dtype:
                    att_slice = att_slice.to(o_proj.weight.dtype)
                out_emb = o_proj(att_slice)
                out_emb = out_emb + hs
                after_res = out_emb.clone()
                out_emb = post_ln(out_emb)
                out_emb = mlp(out_emb)
                out_emb = out_emb + after_res
                outputs_embeds.append(out_emb)
                start = end if len(att_outputs) == 1 else 0

            inputs_embeds = outputs_embeds

        # Final norms
        prefix_out = self.vlm_norm(inputs_embeds[0])
        suffix_out = self.exp_norm(inputs_embeds[1])
        suffix_out = suffix_out[:, -CHUNK_SIZE:]
        return suffix_out


def get_inputs():
    B = 1
    prefix_len = 113
    suffix_len = CHUNK_SIZE
    total_len = prefix_len + suffix_len

    prefix_embs = torch.randn(B, prefix_len, TEXT_HIDDEN_SIZE)
    suffix_embs = torch.randn(B, suffix_len, EXPERT_HIDDEN_SIZE)

    pad_masks = torch.ones(B, total_len, dtype=torch.bool)
    att_masks = torch.zeros(B, total_len, dtype=torch.bool)
    att_masks[:, prefix_len - 1] = True
    att_masks[:, prefix_len:] = True

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks

    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    return [prefix_embs, suffix_embs, att_2d_masks, position_ids]


def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, CHUNK_SIZE, EXPERT_HIDDEN_SIZE)]

def run_tests():
    try:
        model = RefactoredModel()
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
    sys.exit(0 if run_tests() else 1)
