"""
Component: VLM+Expert Transformer (16 layers with alternating self/cross attention)
Abstraction Level: layer
Parent: VLAFlowMatching (root)

Operations: 16x (RMSNorm + QKV projection + RoPE + attention + o_proj + residual +
            RMSNorm + SwiGLU MLP + residual), final RMSNorm x2

Input Shapes:
  - prefix_embs: [B, prefix_len, 960] float32  (prefix_len=113 for 1 image + 48 lang + 1 state)
  - suffix_embs: [B, 50, 720] float32
  - att_2d_masks: [B, prefix_len+50, prefix_len+50] bool
  - position_ids: [B, prefix_len+50] int64

Output Shapes:
  - suffix_out: [B, 50, 720] float32

Weight Shapes:
  - VLM: 16 Llama layers (960 hidden, 15 heads, 5 KV heads, 2560 intermediate)
  - Expert: 16 Llama layers (720 hidden, 15 heads, 5 KV heads, 2048 intermediate)
  - VLM text_model.embed_tokens: not used here (embedding done elsewhere)
  - VLM text_model.norm: RMSNorm(960)
  - Expert norm: RMSNorm(720)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Config
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


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=TEXT_RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def apply_rope(x, positions, max_wavelength=10_000):
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


def eager_attention_forward(attention_mask, batch_size, head_dim, query_states, key_states, value_states,
                            num_attention_heads=TEXT_NUM_HEADS, num_key_value_heads=TEXT_NUM_KV_HEADS):
    num_key_value_groups = num_attention_heads // num_key_value_heads
    sequence_length = key_states.shape[1]
    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    ).reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    ).reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
    query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
    key_states = key_states.to(dtype=torch.float32).transpose(1, 2)
    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights *= head_dim ** -0.5
    att_weights = att_weights.to(dtype=torch.float32)
    big_neg = torch.finfo(att_weights.dtype).min
    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
    probs = F.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)
    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
    att_output = att_output.permute(0, 2, 1, 3)
    att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
    return att_output


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_eps):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_eps)


class Model(nn.Module):
    """VLM+Expert Transformer: 16 layers with alternating self/cross attention."""

    def __init__(self):
        super().__init__()
        # VLM layers
        self.vlm_layers = nn.ModuleList([
            LlamaDecoderLayer(TEXT_HIDDEN_SIZE, TEXT_NUM_HEADS, TEXT_NUM_KV_HEADS,
                              TEXT_HEAD_DIM, TEXT_INTERMEDIATE_SIZE, TEXT_RMS_NORM_EPS)
            for _ in range(NUM_VLM_LAYERS)
        ])
        self.vlm_norm = RMSNorm(TEXT_HIDDEN_SIZE, eps=TEXT_RMS_NORM_EPS)

        # Expert layers
        self.expert_layers = nn.ModuleList([
            LlamaDecoderLayer(EXPERT_HIDDEN_SIZE, EXPERT_NUM_HEADS, EXPERT_NUM_KV_HEADS,
                              TEXT_HEAD_DIM, EXPERT_INTERMEDIATE_SIZE, TEXT_RMS_NORM_EPS)
            for _ in range(NUM_VLM_LAYERS)
        ])
        self.expert_norm = RMSNorm(EXPERT_HIDDEN_SIZE, eps=TEXT_RMS_NORM_EPS)

        # Replace KV projections for cross-attention layers (odd layers)
        vlm_kv_dim = TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM  # 320
        expert_kv_dim = EXPERT_NUM_KV_HEADS * TEXT_HEAD_DIM  # 320
        for layer_idx in range(NUM_VLM_LAYERS):
            if SELF_ATTN_EVERY_N_LAYERS > 0 and layer_idx % SELF_ATTN_EVERY_N_LAYERS == 0:
                continue
            self.expert_layers[layer_idx].self_attn.k_proj = nn.Linear(vlm_kv_dim, expert_kv_dim, bias=False)
            self.expert_layers[layer_idx].self_attn.v_proj = nn.Linear(vlm_kv_dim, expert_kv_dim, bias=False)

    def get_model_layers(self):
        vlm_layers_list = []
        expert_layers_list = []
        for i in range(NUM_VLM_LAYERS):
            vlm_layers_list.append(self.vlm_layers[i])
            expert_layers_list.append(self.expert_layers[i])
        return [vlm_layers_list, expert_layers_list]

    def forward_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids,
                           attention_mask, batch_size, head_dim):
        query_states, key_states, value_states = [], [], []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape))
            key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape))
            value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape))

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        query_states = apply_rope(query_states, _position_ids)
        key_states = apply_rope(key_states, _position_ids)
        att_output = eager_attention_forward(_attention_mask, batch_size, head_dim, query_states, key_states, value_states)
        return [att_output]

    def forward_cross_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids,
                                 attention_mask, batch_size, head_dim):
        att_outputs = []
        p_len = inputs_embeds[0].shape[1]
        pos_id = position_ids[:, :p_len]
        exp_pos_id = position_ids[:, p_len:]
        prefix_mask = attention_mask[:, :p_len, :p_len]

        layer = model_layers[0][layer_idx]
        hidden_states = layer.input_layernorm(inputs_embeds[0])
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
        value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

        query_states = apply_rope(query_state, pos_id)
        key_states = apply_rope(key_state, pos_id)
        att_output = eager_attention_forward(prefix_mask, batch_size, head_dim, query_states, key_states, value_states)
        att_outputs.append(att_output)

        # Expert cross-attention
        expert_layer = model_layers[1][layer_idx]
        expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
        expert_input_shape = expert_hidden_states.shape[:-1]
        expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)
        expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
        expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

        _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(*key_states.shape[:2], -1)
        expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(*_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim)
        _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(*value_states.shape[:2], -1)
        expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(*_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim)

        exp_pos_id = exp_pos_id - torch.min(exp_pos_id, dim=1, keepdim=True).values
        expert_attention_mask = attention_mask[:, -inputs_embeds[1].shape[1]:, :expert_key_states.shape[1]:]
        expert_query_states = apply_rope(expert_query_state, exp_pos_id)
        att_output = eager_attention_forward(expert_attention_mask, batch_size, head_dim,
                                             expert_query_states, expert_key_states, expert_value_states)
        att_outputs.append(att_output)
        return att_outputs

    def forward(self, prefix_embs, suffix_embs, att_2d_masks, position_ids):
        model_layers = self.get_model_layers()
        inputs_embeds = [prefix_embs, suffix_embs]
        batch_size = prefix_embs.shape[0]
        head_dim = TEXT_HEAD_DIM

        for layer_idx in range(NUM_VLM_LAYERS):
            if SELF_ATTN_EVERY_N_LAYERS > 0 and layer_idx % SELF_ATTN_EVERY_N_LAYERS == 0:
                att_outputs = self.forward_attn_layer(
                    model_layers, inputs_embeds, layer_idx, position_ids,
                    att_2d_masks, batch_size, head_dim)
            else:
                att_outputs = self.forward_cross_attn_layer(
                    model_layers, inputs_embeds, layer_idx, position_ids,
                    att_2d_masks, batch_size, head_dim)

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)
                    out_emb += after_first_residual
                    outputs_embeds.append(out_emb)
                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)
            inputs_embeds = outputs_embeds

        # Final norms
        prefix_out = self.vlm_norm(inputs_embeds[0])
        suffix_out = self.expert_norm(inputs_embeds[1])
        suffix_out = suffix_out[:, -CHUNK_SIZE:]
        return suffix_out


def get_inputs():
    B = 1
    prefix_len = 113  # 64 img + 48 lang + 1 state
    suffix_len = CHUNK_SIZE  # 50
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
            print(f"Input shapes: prefix={inputs[0].shape}, suffix={inputs[1].shape}")
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
