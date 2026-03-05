"""Debug: compare wrapper vs original at single layer level."""
import sys, copy
from pathlib import Path
import torch
import torch.nn as nn

from transformers import AutoConfig, SmolVLMForConditionalGeneration, AutoModel

config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
vlm = SmolVLMForConditionalGeneration(config=config)
vlm.model.text_model.layers = vlm.model.text_model.layers[:16]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "level_3_model"))
from smolvla import Model, apply_rope, eager_attention_forward

wrapper = Model()

# Build expert
lm_expert_config = copy.deepcopy(config.text_config)
lm_expert_config.hidden_size = int(960 * 0.75)
def get_intermediate_size(hd, m=4, mul=256):
    hd = int(2*hd/3); hd = int(m*hd); hd = mul*((hd+mul-1)//mul); return hd
lm_expert_config.intermediate_size = get_intermediate_size(lm_expert_config.hidden_size)
lm_expert_config.num_hidden_layers = 16
lm_expert = AutoModel.from_config(lm_expert_config)
lm_expert.embed_tokens = None
for li in range(len(lm_expert.layers)):
    if li % 2 == 0: continue
    lm_expert.layers[li].self_attn.k_proj = nn.Linear(320, 320, bias=False)
    lm_expert.layers[li].self_attn.v_proj = nn.Linear(320, 320, bias=False)

# Transfer weights
orig_combined = {}
for key, val in vlm.model.vision_model.state_dict().items():
    orig_combined[f"vlm_with_expert.vision_encoder.{key}"] = val
for key, val in vlm.model.connector.state_dict().items():
    if key == "modality_projection.proj.weight":
        orig_combined["vlm_with_expert.connector.proj.weight"] = val
for key, val in vlm.model.text_model.embed_tokens.state_dict().items():
    orig_combined[f"vlm_with_expert.text_model.embed_tokens.{key}"] = val
for i in range(16):
    for key, val in vlm.model.text_model.layers[i].state_dict().items():
        orig_combined[f"vlm_with_expert.text_model.layers.{i}.{key}"] = val
for key, val in vlm.model.text_model.norm.state_dict().items():
    orig_combined[f"vlm_with_expert.text_model.norm.{key}"] = val
for i in range(16):
    for key, val in lm_expert.layers[i].state_dict().items():
        orig_combined[f"vlm_with_expert.lm_expert.layers.{i}.{key}"] = val
for key, val in lm_expert.norm.state_dict().items():
    orig_combined[f"vlm_with_expert.lm_expert.norm.{key}"] = val

wrap_sd = wrapper.state_dict()
for k in wrap_sd:
    if k in orig_combined and orig_combined[k].shape == wrap_sd[k].shape:
        wrap_sd[k] = orig_combined[k].clone()
wrapper.load_state_dict(wrap_sd)
wrapper.eval(); vlm.eval(); lm_expert.eval()

# Create synthetic inputs that are identical for both
torch.manual_seed(42)
B, prefix_len, suffix_len = 1, 113, 50
hidden_vlm = 960
hidden_exp = 720
head_dim = 64

prefix_embs = torch.randn(B, prefix_len, hidden_vlm)
suffix_embs = torch.randn(B, suffix_len, hidden_exp)

# Create masks similar to real model
pad_masks = torch.ones(B, prefix_len + suffix_len, dtype=torch.bool)
att_masks = torch.zeros(B, prefix_len + suffix_len, dtype=torch.bool)
att_masks[:, prefix_len - 1] = True  # state token
att_masks[:, prefix_len:] = True  # suffix tokens

from smolvla import make_att_2d_masks
att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
position_ids = torch.cumsum(pad_masks, dim=1) - 1

# Compare layer 0 (self-attention layer)
# Wrapper
w_vlm_layer = wrapper.vlm_with_expert.text_model.layers[0]
w_exp_layer = wrapper.vlm_with_expert.lm_expert.layers[0]
# Original
o_vlm_layer = vlm.model.text_model.layers[0]
o_exp_layer = lm_expert.layers[0]

# Verify weights match
for name in w_vlm_layer.state_dict():
    w_val = w_vlm_layer.state_dict()[name]
    o_val = o_vlm_layer.state_dict()[name]
    diff = (w_val - o_val).abs().max().item()
    if diff > 0:
        print(f"  VLM layer 0 weight diff: {name} = {diff:.2e}")

for name in w_exp_layer.state_dict():
    w_val = w_exp_layer.state_dict()[name]
    o_val = o_exp_layer.state_dict()[name]
    diff = (w_val - o_val).abs().max().item()
    if diff > 0:
        print(f"  Expert layer 0 weight diff: {name} = {diff:.2e}")

print("Weights verified identical.")

# Test layer 0 self-attention step
with torch.no_grad():
    # VLM layernorm
    w_hn = w_vlm_layer.input_layernorm(prefix_embs)
    o_hn = o_vlm_layer.input_layernorm(prefix_embs)
    print(f"VLM RMSNorm diff: {(w_hn - o_hn).abs().max().item():.2e}")

    # Expert layernorm
    w_ehn = w_exp_layer.input_layernorm(suffix_embs)
    o_ehn = o_exp_layer.input_layernorm(suffix_embs)
    print(f"Expert RMSNorm diff: {(w_ehn - o_ehn).abs().max().item():.2e}")

    # QKV projections VLM
    w_q = w_vlm_layer.self_attn.q_proj(w_hn)
    o_q = o_vlm_layer.self_attn.q_proj(o_hn)
    print(f"VLM Q proj diff: {(w_q - o_q).abs().max().item():.2e}")

    # Full self-attn forward for layer 0
    # In wrapper: both prefix and suffix go through self-attn
    inputs_w = [prefix_embs.clone(), suffix_embs.clone()]
    inputs_o = [prefix_embs.clone(), suffix_embs.clone()]

    # Wrapper forward layer 0
    model_layers_w = wrapper.vlm_with_expert.get_model_layers()

    # Self-attn layer: both streams participate
    # Step 1: compute QKV for both streams
    qs_w, ks_w, vs_w = [], [], []
    for i, hs in enumerate(inputs_w):
        layer = model_layers_w[i][0]
        if hs is None or layer is None: continue
        hn = layer.input_layernorm(hs)
        shape = (*hn.shape[:-1], -1, layer.self_attn.head_dim)
        hn = hn.to(layer.self_attn.q_proj.weight.dtype)
        qs_w.append(layer.self_attn.q_proj(hn).view(shape))
        ks_w.append(layer.self_attn.k_proj(hn).view(shape))
        vs_w.append(layer.self_attn.v_proj(hn).view(shape))

    all_q_w = torch.cat(qs_w, dim=1)
    all_k_w = torch.cat(ks_w, dim=1)
    all_v_w = torch.cat(vs_w, dim=1)

    # Same for original
    model_layers_o = [list(vlm.model.text_model.layers), list(lm_expert.layers)]
    qs_o, ks_o, vs_o = [], [], []
    for i, hs in enumerate(inputs_o):
        layer = model_layers_o[i][0]
        if hs is None or layer is None: continue
        hn = layer.input_layernorm(hs)
        shape = (*hn.shape[:-1], -1, layer.self_attn.head_dim)
        hn = hn.to(layer.self_attn.q_proj.weight.dtype)
        qs_o.append(layer.self_attn.q_proj(hn).view(shape))
        ks_o.append(layer.self_attn.k_proj(hn).view(shape))
        vs_o.append(layer.self_attn.v_proj(hn).view(shape))

    all_q_o = torch.cat(qs_o, dim=1)
    all_k_o = torch.cat(ks_o, dim=1)
    all_v_o = torch.cat(vs_o, dim=1)

    print(f"QKV concat diff: Q={( all_q_w - all_q_o).abs().max():.2e} K={( all_k_w - all_k_o).abs().max():.2e} V={( all_v_w - all_v_o).abs().max():.2e}")

    # Apply RoPE and attention
    seq_len_q = all_q_w.shape[1]
    _pos = position_ids[:, :seq_len_q]
    _mask = att_2d_masks[:, :seq_len_q, :seq_len_q]

    all_q_w = apply_rope(all_q_w, _pos)
    all_k_w = apply_rope(all_k_w, _pos)
    all_q_o = apply_rope(all_q_o, _pos)
    all_k_o = apply_rope(all_k_o, _pos)

    print(f"Post-RoPE diff: Q={( all_q_w - all_q_o).abs().max():.2e} K={( all_k_w - all_k_o).abs().max():.2e}")

    att_w = eager_attention_forward(_mask, 1, head_dim, all_q_w, all_k_w, all_v_w)
    att_o = eager_attention_forward(_mask, 1, head_dim, all_q_o, all_k_o, all_v_o)
    print(f"Attention output diff: {(att_w - att_o).abs().max():.2e}")

    # Post-attention
    for i, hs in enumerate(inputs_w):
        layer = model_layers_w[i][0]
        if hs is None or layer is None: continue
        end = hs.shape[1] if i == 0 else hs.shape[1]
        start_idx = 0 if i == 0 else prefix_len
        a_slice_w = att_w[:, start_idx:start_idx+hs.shape[1]]
        a_slice_o = att_o[:, start_idx:start_idx+hs.shape[1]]
        print(f"  Stream {i}: att_slice diff = {(a_slice_w - a_slice_o).abs().max():.2e}")

        o_proj_w = layer.self_attn.o_proj(a_slice_w)
        o_proj_o = model_layers_o[i][0].self_attn.o_proj(a_slice_o)
        print(f"  Stream {i}: o_proj diff = {(o_proj_w - o_proj_o).abs().max():.2e}")

    print("\nAll checks above should be 0.00e+00 if implementation is correct.")
