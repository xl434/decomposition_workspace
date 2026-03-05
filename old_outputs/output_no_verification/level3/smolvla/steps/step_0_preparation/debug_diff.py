"""Debug script to compare wrapper vs original at transformer layer level."""
import sys, copy, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, SmolVLMForConditionalGeneration, AutoModel

config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
vlm = SmolVLMForConditionalGeneration(config=config)
vlm.model.text_model.layers = vlm.model.text_model.layers[:16]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "level_3_model"))
from smolvla import Model, get_inputs, make_att_2d_masks, apply_rope, eager_attention_forward

wrapper = Model()

# Build expert
lm_expert_config = copy.deepcopy(config.text_config)
lm_expert_config.hidden_size = int(960 * 0.75)
def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim
lm_expert_config.intermediate_size = get_intermediate_size(lm_expert_config.hidden_size)
lm_expert_config.num_hidden_layers = 16
lm_expert = AutoModel.from_config(lm_expert_config)
lm_expert.embed_tokens = None
for layer_idx in range(len(lm_expert.layers)):
    if layer_idx % 2 == 0:
        continue
    lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(320, 320, bias=False)
    lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(320, 320, bias=False)

# Transfer ALL weights to wrapper
orig_combined = {}
for key, val in vlm.model.vision_model.state_dict().items():
    orig_combined[f"vlm_with_expert.vision_encoder.{key}"] = val
for key, val in vlm.model.connector.state_dict().items():
    if key == "modality_projection.proj.weight":
        orig_combined["vlm_with_expert.connector.proj.weight"] = val
    else:
        orig_combined[f"vlm_with_expert.connector.{key}"] = val
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
mapped = 0
for k in wrap_sd:
    if k in orig_combined and orig_combined[k].shape == wrap_sd[k].shape:
        wrap_sd[k] = orig_combined[k].clone()
        mapped += 1
wrapper.load_state_dict(wrap_sd)
wrapper.eval()
vlm.eval()
lm_expert.eval()
print(f"Mapped {mapped}/{len(wrap_sd)}")

# Use same inputs and run BOTH forwards with SAME prefix/suffix embeddings
torch.manual_seed(42)
inputs = get_inputs()
images, img_masks, lang_tokens, lang_masks, state, actions, noise, time = inputs

with torch.no_grad():
    # Wrapper full forward
    wrap_out = wrapper(*inputs)

    # Now manual forward using WRAPPER embeddings but ORIGINAL weights
    # First get embeddings from wrapper
    w_prefix_embs, w_prefix_pad, w_prefix_att = wrapper.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state
    )
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    w_suffix_embs, w_suffix_pad, w_suffix_att = wrapper.embed_suffix(x_t, time)

    pad_masks = torch.cat([w_prefix_pad, w_suffix_pad], dim=1)
    att_masks = torch.cat([w_prefix_att, w_suffix_att], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    # Forward through wrapper vlm_with_expert
    (_, w_suffix_out), _ = wrapper.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[w_prefix_embs, w_suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )

    # Forward through original components using the SAME embeddings
    model_layers = [list(vlm.model.text_model.layers), list(lm_expert.layers)]
    inputs_embeds_o = [w_prefix_embs.clone(), w_suffix_embs.clone()]
    batch_size = 1
    head_dim = 64

    for layer_idx in range(16):
        if layer_idx % 2 == 0:
            # Self-attention layer
            query_states_list = []
            key_states_list = []
            value_states_list = []
            for i_m, hs in enumerate(inputs_embeds_o):
                layer = model_layers[i_m][layer_idx]
                if hs is None or layer is None:
                    continue
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

            seq_len_q = all_q.shape[1]
            if seq_len_q < position_ids.shape[1]:
                _pos = position_ids[:, :seq_len_q]
                _mask = att_2d_masks[:, :seq_len_q, :seq_len_q]
            else:
                _pos = position_ids
                _mask = att_2d_masks

            all_q = apply_rope(all_q, _pos)
            all_k = apply_rope(all_k, _pos)
            att_output = eager_attention_forward(_mask, batch_size, head_dim, all_q, all_k, all_v)
            att_outputs = [att_output]
        else:
            # Cross-attention layer
            att_outputs = []
            p_len = inputs_embeds_o[0].shape[1]
            pos_id = position_ids[:, :p_len]
            exp_pos_id = position_ids[:, p_len:]
            prefix_mask = att_2d_masks[:, :p_len, :p_len]

            layer = model_layers[0][layer_idx]
            hs_norm = layer.input_layernorm(inputs_embeds_o[0])
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
            e_hs = expert_layer.input_layernorm(inputs_embeds_o[1])
            e_shape = e_hs.shape[:-1]
            e_hidden_shape = (*e_shape, -1, expert_layer.self_attn.head_dim)
            e_hs = e_hs.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            e_q = expert_layer.self_attn.q_proj(e_hs).view(e_hidden_shape)
            _k = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            e_k = expert_layer.self_attn.k_proj(_k).view(
                *_k.shape[:-1], -1, expert_layer.self_attn.head_dim
            )
            _v = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            e_v = expert_layer.self_attn.v_proj(_v).view(
                *_v.shape[:-1], -1, expert_layer.self_attn.head_dim
            )
            exp_pos_id = exp_pos_id - torch.min(exp_pos_id, dim=1, keepdim=True).values
            e_mask = att_2d_masks[
                :, -inputs_embeds_o[1].shape[1]:, :e_k.shape[1]:
            ]
            e_q = apply_rope(e_q, exp_pos_id)
            a_out = eager_attention_forward(e_mask, batch_size, head_dim, e_q, e_k, e_v)
            att_outputs.append(a_out)

        # Post-attention
        outputs_embeds = []
        start = 0
        for i_m, hs in enumerate(inputs_embeds_o):
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
        inputs_embeds_o = outputs_embeds

    # Final norm
    models_list = [vlm.model.text_model, lm_expert]
    final_outputs = []
    for i_m, hs in enumerate(inputs_embeds_o):
        if hs is not None:
            o = models_list[i_m].norm(hs)
            final_outputs.append(o)
        else:
            final_outputs.append(None)

    o_suffix_out = final_outputs[1]

    diff = (o_suffix_out.float() - w_suffix_out.float()).abs().max().item()
    print(f"Transformer output diff (with SAME inputs, SAME weights): {diff:.2e}")

    if diff < 1e-5:
        print("=> Transformer layers are IDENTICAL. Diff comes from vision encoder.")
    else:
        print("=> BUG in transformer layer implementation!")
