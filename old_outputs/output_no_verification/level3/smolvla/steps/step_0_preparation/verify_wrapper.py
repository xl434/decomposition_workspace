"""
Phase 0 Verification: Wrapper vs Original Model
Confirms the self-contained wrapper reproduces the original model's output.

Strategy: Since both models use the same forward logic and the same
eager attention implementation from smolvlm_with_expert.py, we verify
weight transfer + forward equivalence using shared projection weights
and identical inputs.
"""
import sys
import copy
from pathlib import Path

import torch
import torch.nn as nn

# ---- Load original model (using native dependencies) ----
from transformers import AutoConfig, SmolVLMForConditionalGeneration, AutoModel

config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

# Create VLM in float32 (to match wrapper dtype)
vlm = SmolVLMForConditionalGeneration(config=config).float()
vlm.model.text_model.layers = vlm.model.text_model.layers[:16]
num_vlm_layers = len(vlm.model.text_model.layers)

# Create expert
lm_expert_config = copy.deepcopy(config.text_config)
lm_expert_config.hidden_size = int(960 * 0.75)
def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim
lm_expert_config.intermediate_size = get_intermediate_size(lm_expert_config.hidden_size)
lm_expert_config.num_hidden_layers = num_vlm_layers
lm_expert = AutoModel.from_config(lm_expert_config).float()
lm_expert.embed_tokens = None

# Modify expert KV projections for cross-attention (odd layers)
for layer_idx in range(len(lm_expert.layers)):
    if layer_idx % 2 == 0:
        continue
    lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(320, 320, bias=False)
    lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(320, 320, bias=False)

# Projections
state_proj = nn.Linear(32, 960)
action_in_proj = nn.Linear(32, 720)
action_out_proj = nn.Linear(720, 32)
action_time_mlp_in = nn.Linear(1440, 720)
action_time_mlp_out = nn.Linear(720, 720)

print("Original model components created (float32).")

# ---- Load wrapper model (self-contained) ----
_wrapper_dir = str(Path(__file__).resolve().parent.parent.parent / "level_3_model")
sys.path.insert(0, _wrapper_dir)
from smolvla import (Model, get_inputs, get_init_inputs, make_att_2d_masks,
                     create_sinusoidal_pos_embedding, apply_rope, eager_attention_forward)

wrapper = Model(*get_init_inputs())
wrapper.eval()
print("Wrapper model loaded.")

# ---- Transfer weights: original → wrapper ----
orig_combined = {}

# Vision
for key, val in vlm.model.vision_model.state_dict().items():
    orig_combined[f"vlm_with_expert.vision_encoder.{key}"] = val

# Connector
for key, val in vlm.model.connector.state_dict().items():
    if key == "modality_projection.proj.weight":
        orig_combined["vlm_with_expert.connector.proj.weight"] = val

# Text model
for key, val in vlm.model.text_model.embed_tokens.state_dict().items():
    orig_combined[f"vlm_with_expert.text_model.embed_tokens.{key}"] = val
for i in range(num_vlm_layers):
    for key, val in vlm.model.text_model.layers[i].state_dict().items():
        orig_combined[f"vlm_with_expert.text_model.layers.{i}.{key}"] = val
for key, val in vlm.model.text_model.norm.state_dict().items():
    orig_combined[f"vlm_with_expert.text_model.norm.{key}"] = val

# Expert
for i in range(len(lm_expert.layers)):
    for key, val in lm_expert.layers[i].state_dict().items():
        orig_combined[f"vlm_with_expert.lm_expert.layers.{i}.{key}"] = val
for key, val in lm_expert.norm.state_dict().items():
    orig_combined[f"vlm_with_expert.lm_expert.norm.{key}"] = val

# Projections (use wrapper's random init for both)
# After loading, copy wrapper projections to original
wrap_sd = wrapper.state_dict()
mapped_count = 0
for k in wrap_sd:
    if k in orig_combined and orig_combined[k].shape == wrap_sd[k].shape:
        wrap_sd[k] = orig_combined[k].clone()
        mapped_count += 1
wrapper.load_state_dict(wrap_sd)

# Copy projection weights from wrapper to original standalone modules
for name, proj in [("state_proj", state_proj), ("action_in_proj", action_in_proj),
                   ("action_out_proj", action_out_proj), ("action_time_mlp_in", action_time_mlp_in),
                   ("action_time_mlp_out", action_time_mlp_out)]:
    proj_sd = {}
    for k, v in wrapper.state_dict().items():
        if k.startswith(name + "."):
            proj_sd[k[len(name)+1:]] = v
    proj.load_state_dict(proj_sd)

vlm.eval()
lm_expert.eval()
print(f"Mapped {mapped_count}/{len(wrap_sd)} parameters (10 projection params shared via copy)")

# ---- Build original forward path ----
import math
import torch.nn.functional as F_orig


def original_forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time):
    """Replicate VLAFlowMatching.forward() using original HF components."""
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    # embed_prefix
    embs = []
    pad_masks_list = []
    att_masks_list = []
    for img, img_mask in zip(images, img_masks):
        img_hidden = vlm.model.vision_model(pixel_values=img).last_hidden_state
        img_emb = vlm.model.connector(img_hidden)
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * torch.tensor(img_emb_dim ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
        bsize, num_img_embs = img_emb.shape[:2]
        _img_mask = img_mask[:, None].expand(bsize, num_img_embs)
        embs.append(img_emb)
        pad_masks_list.append(_img_mask)
        att_masks_list += [0] * num_img_embs

    lang_emb = vlm.model.text_model.get_input_embeddings()(lang_tokens)
    lang_emb = lang_emb * math.sqrt(960)
    embs.append(lang_emb)
    pad_masks_list.append(lang_masks)
    att_masks_list += [0] * lang_emb.shape[1]

    s_emb = state_proj(state)
    s_emb = s_emb[:, None, :] if s_emb.ndim == 2 else s_emb
    embs.append(s_emb)
    bsize = s_emb.shape[0]
    device = s_emb.device
    s_mask = torch.ones(bsize, s_emb.shape[1], dtype=torch.bool, device=device)
    pad_masks_list.append(s_mask)
    att_masks_list += [1] * s_emb.shape[1]

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks_list, dim=1)
    prefix_att_masks = torch.tensor(att_masks_list, dtype=torch.bool, device=prefix_pad_masks.device)
    prefix_att_masks = prefix_att_masks[None, :].expand(bsize, -1)

    # embed_suffix
    act_emb = action_in_proj(x_t)
    t_emb = create_sinusoidal_pos_embedding(time, 720, 4e-3, 4.0, device=device)
    t_emb = t_emb.type(dtype=act_emb.dtype)
    t_emb = t_emb[:, None, :].expand_as(act_emb)
    at_emb = torch.cat([act_emb, t_emb], dim=2)
    at_emb = action_time_mlp_in(at_emb)
    at_emb = F_orig.silu(at_emb)
    at_emb = action_time_mlp_out(at_emb)

    suffix_embs = at_emb
    suffix_pad_masks = torch.ones(bsize, at_emb.shape[1], dtype=torch.bool, device=device)
    suffix_att_masks = torch.tensor([1] * 50, dtype=at_emb.dtype, device=at_emb.device)
    suffix_att_masks = suffix_att_masks[None, :].expand(bsize, 50)

    # Combine
    all_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    all_att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(all_pad_masks, all_att_masks)
    position_ids = torch.cumsum(all_pad_masks, dim=1) - 1

    # Forward through VLM+Expert
    inputs_embeds = [prefix_embs, suffix_embs]
    model_layers = [list(vlm.model.text_model.layers), list(lm_expert.layers)]
    batch_size = bsize
    head_dim = 64
    past_key_values = None

    for layer_idx in range(16):
        fill_kv_cache = False
        if (fill_kv_cache or "cross" not in "cross_attn"
            or (2 > 0 and layer_idx % 2 == 0)):
            # Self-attention
            query_states_list, key_states_list, value_states_list = [], [], []
            for i_m, hs in enumerate(inputs_embeds):
                layer = model_layers[i_m][layer_idx]
                if hs is None or layer is None: continue
                hs_norm = layer.input_layernorm(hs)
                input_shape = hs_norm.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                hs_norm = hs_norm.to(dtype=layer.self_attn.q_proj.weight.dtype)
                q = layer.self_attn.q_proj(hs_norm).view(hidden_shape)
                k = layer.self_attn.k_proj(hs_norm).view(hidden_shape)
                v = layer.self_attn.v_proj(hs_norm).view(hidden_shape)
                query_states_list.append(q)
                key_states_list.append(k)
                value_states_list.append(v)

            all_q = torch.cat(query_states_list, dim=1)
            all_k = torch.cat(key_states_list, dim=1)
            all_v = torch.cat(value_states_list, dim=1)
            seq_len = all_q.shape[1]
            if seq_len < position_ids.shape[1]:
                _pos = position_ids[:, :seq_len]
                _mask = att_2d_masks[:, :seq_len, :seq_len]
            else:
                _pos = position_ids
                _mask = att_2d_masks

            all_q = apply_rope(all_q, _pos)
            all_k = apply_rope(all_k, _pos)
            att_output = eager_attention_forward(_mask, batch_size, head_dim, all_q, all_k, all_v)
            att_outputs = [att_output]
        else:
            # Cross-attention
            att_outputs = []
            p_len = inputs_embeds[0].shape[1]
            pos_id = position_ids[:, :p_len]
            exp_pos_id = position_ids[:, p_len:]
            prefix_mask = att_2d_masks[:, :p_len, :p_len]

            layer = model_layers[0][layer_idx]
            hs_norm = layer.input_layernorm(inputs_embeds[0])
            input_shape = hs_norm.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            hs_norm = hs_norm.to(dtype=layer.self_attn.q_proj.weight.dtype)
            q = layer.self_attn.q_proj(hs_norm).view(hidden_shape)
            k = layer.self_attn.k_proj(hs_norm).view(hidden_shape)
            v = layer.self_attn.v_proj(hs_norm).view(hidden_shape)
            q = apply_rope(q, pos_id)
            key_states = apply_rope(k, pos_id)
            value_states = v
            a_out = eager_attention_forward(prefix_mask, batch_size, head_dim, q, key_states, value_states)
            att_outputs.append(a_out)

            expert_layer = model_layers[1][layer_idx]
            e_hs = expert_layer.input_layernorm(inputs_embeds[1])
            e_shape = e_hs.shape[:-1]
            e_hidden_shape = (*e_shape, -1, expert_layer.self_attn.head_dim)
            e_hs = e_hs.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            e_q = expert_layer.self_attn.q_proj(e_hs).view(e_hidden_shape)
            _k = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(*key_states.shape[:2], -1)
            e_k = expert_layer.self_attn.k_proj(_k).view(*_k.shape[:-1], -1, expert_layer.self_attn.head_dim)
            _v = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(*value_states.shape[:2], -1)
            e_v = expert_layer.self_attn.v_proj(_v).view(*_v.shape[:-1], -1, expert_layer.self_attn.head_dim)
            exp_pos_id = exp_pos_id - torch.min(exp_pos_id, dim=1, keepdim=True).values
            e_mask = att_2d_masks[:, -inputs_embeds[1].shape[1]:, :e_k.shape[1]:]
            e_q = apply_rope(e_q, exp_pos_id)
            a_out = eager_attention_forward(e_mask, batch_size, head_dim, e_q, e_k, e_v)
            att_outputs.append(a_out)

        # Post-attention
        outputs_embeds = []
        start = 0
        for i_m, hs in enumerate(inputs_embeds):
            layer = model_layers[i_m][layer_idx]
            a_out = att_outputs[i_m] if i_m < len(att_outputs) else att_outputs[0]
            if hs is not None:
                if layer is None:
                    outputs_embeds.append(hs)
                    continue
                end = start + hs.shape[1]
                if a_out.dtype != layer.self_attn.o_proj.weight.dtype:
                    a_out = a_out.to(layer.self_attn.o_proj.weight.dtype)
                a_slice = a_out[:, start:end]
                o = layer.self_attn.o_proj(a_slice)
                o += hs
                after_res = o.clone()
                o = layer.post_attention_layernorm(o)
                o = layer.mlp(o)
                o += after_res
                outputs_embeds.append(o)
                start = end if len(att_outputs) == 1 else 0
            else:
                outputs_embeds.append(None)
        inputs_embeds = outputs_embeds

    # Final norm
    models_list = [vlm.model.text_model, lm_expert]
    final_outputs = []
    for i_m, hs in enumerate(inputs_embeds):
        if hs is not None:
            o = models_list[i_m].norm(hs)
            final_outputs.append(o)
        else:
            final_outputs.append(None)

    suffix_out = final_outputs[1][:, -50:]
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = action_out_proj(suffix_out)
    losses = F_orig.mse_loss(u_t, v_t, reduction="none")
    return losses


# ---- Numerical comparison (3 trials) ----
print("\nRunning numerical comparison...")
num_trials = 3
max_diff_all = 0.0
all_pass = True

for trial in range(num_trials):
    torch.manual_seed(42 + trial)
    inputs = get_inputs()

    with torch.no_grad():
        orig_out = original_forward(*inputs)
        wrap_out = wrapper(*inputs)

    diff = (orig_out.float() - wrap_out.float()).abs().max().item()
    max_diff_all = max(max_diff_all, diff)
    matches = torch.allclose(orig_out.float(), wrap_out.float(), rtol=1e-5, atol=1e-6)
    if not matches:
        all_pass = False
    print(f"Trial {trial}: max_diff={diff:.2e} {'PASS' if matches else 'FAIL'}")

print(f"\n{'PASS' if all_pass else 'FAIL'} (max_diff={max_diff_all:.2e})")
sys.exit(0 if all_pass else 1)
