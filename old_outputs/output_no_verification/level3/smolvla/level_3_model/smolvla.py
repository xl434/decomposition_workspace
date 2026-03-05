"""
Component: SmolVLA VLAFlowMatching
Abstraction Level: model
Parent: root
Children: [vision_encoder, connector, vlm_text_layers, expert_layers, embedding, projections, attention_logic]

Operations: Vision encoding (SigLIP), pixel shuffle connector, language embedding,
            state projection, action projection, time embedding, sinusoidal positional encoding,
            RoPE, self-attention, cross-attention, MLP (SwiGLU), RMSNorm, LayerNorm,
            flow matching loss (MSE)

Input Shapes:
  - images: list of [B, 3, 512, 512] float32 tensors (1 image)
  - img_masks: list of [B] bool tensors (1 mask)
  - lang_tokens: [B, 48] int64
  - lang_masks: [B, 48] bool
  - state: [B, 32] float32
  - actions: [B, 50, 32] float32
  - noise: [B, 50, 32] float32 (optional)
  - time: [B] float32 (optional)

Output Shapes:
  - losses: [B, 50, 32] float32

Weight Shapes: (see module hierarchy below)
  - Vision encoder: ~7M params (12 SigLIP layers)
  - VLM text model: ~172M params (16 Llama layers + embeddings)
  - Expert model: ~77M params (16 Llama layers)
  - Projections: state_proj, action_in_proj, action_out_proj, action_time_mlp_in/out
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Configuration Constants (SmolVLM2-500M-Video-Instruct)
# ============================================================

# Vision Config (SigLIP)
VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_HEAD_DIM = 64  # 768 // 12
VISION_INTERMEDIATE_SIZE = 3072
VISION_NUM_LAYERS = 12
VISION_IMAGE_SIZE = 512
VISION_PATCH_SIZE = 16
VISION_NUM_CHANNELS = 3
VISION_LAYER_NORM_EPS = 1e-6
VISION_NUM_PATCHES_PER_SIDE = VISION_IMAGE_SIZE // VISION_PATCH_SIZE  # 32
VISION_NUM_PATCHES = VISION_NUM_PATCHES_PER_SIDE ** 2  # 1024

# Text Config (LLama-based)
TEXT_HIDDEN_SIZE = 960
TEXT_NUM_HEADS = 15
TEXT_NUM_KV_HEADS = 5
TEXT_HEAD_DIM = 64
TEXT_INTERMEDIATE_SIZE = 2560
TEXT_NUM_LAYERS_FULL = 32  # full model
TEXT_VOCAB_SIZE = 49280
TEXT_RMS_NORM_EPS = 1e-5
TEXT_MAX_POS_EMBEDDINGS = 8192
TEXT_ROPE_THETA = 100000

# Top-level
SCALE_FACTOR = 4  # pixel shuffle factor
CONNECTOR_INPUT_DIM = VISION_HIDDEN_SIZE * (SCALE_FACTOR ** 2)  # 768 * 16 = 12288
CONNECTOR_OUTPUT_DIM = TEXT_HIDDEN_SIZE  # 960

# SmolVLA-specific Config
NUM_VLM_LAYERS = 16  # reduced from 32
EXPERT_WIDTH_MULTIPLIER = 0.75
EXPERT_HIDDEN_SIZE = int(TEXT_HIDDEN_SIZE * EXPERT_WIDTH_MULTIPLIER)  # 720
SELF_ATTN_EVERY_N_LAYERS = 2
ATTENTION_MODE = "cross_attn"

# Expert MLP intermediate size
def _get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

EXPERT_INTERMEDIATE_SIZE = _get_intermediate_size(EXPERT_HIDDEN_SIZE)  # 2048
EXPERT_NUM_HEADS = TEXT_NUM_HEADS  # 15
EXPERT_NUM_KV_HEADS = TEXT_NUM_KV_HEADS  # 5

# Action/State dimensions
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
NUM_STEPS = 10
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0
PREFIX_LENGTH = -1  # no padding
ADD_IMAGE_SPECIAL_TOKENS = False

# Token IDs
FAKE_IMAGE_TOKEN_ID = 49189
GLOBAL_IMAGE_TOKEN_ID = 49152

# Number of image tokens after connector: 1024 / 16 = 64
NUM_IMAGE_TOKENS = VISION_NUM_PATCHES // (SCALE_FACTOR ** 2)  # 64

# Batch size for inputs
BATCH_SIZE = 1
LANG_SEQ_LEN = 48

# ============================================================
# Vision Encoder (SigLIP-based)
# ============================================================

class VisionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = VISION_HIDDEN_SIZE
        self.image_size = VISION_IMAGE_SIZE
        self.patch_size = VISION_PATCH_SIZE
        self.num_patches_per_side = VISION_NUM_PATCHES_PER_SIDE
        self.num_patches = VISION_NUM_PATCHES

        self.patch_embedding = nn.Conv2d(
            in_channels=VISION_NUM_CHANNELS,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(self, pixel_values, patch_attention_mask=None):
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size

        if patch_attention_mask is None:
            patch_attention_mask = torch.ones(
                (batch_size, max_nb_patches_h, max_nb_patches_w),
                dtype=torch.bool, device=pixel_values.device
            )

        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()
            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)
            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class VisionAttention(nn.Module):
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

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output


class VisionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE)
        self.fc2 = nn.Linear(VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate='tanh')
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class VisionEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = VisionAttention()
        self.layer_norm1 = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)
        self.mlp = VisionMLP()
        self.layer_norm2 = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class VisionEncoderBlock(nn.Module):
    """Container for vision encoder layers (matches HF's SmolVLMEncoder)."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([VisionEncoderLayer() for _ in range(VISION_NUM_LAYERS)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = VisionEmbeddings()
        self.encoder = VisionEncoderBlock()
        self.post_layernorm = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


# ============================================================
# Connector (Pixel Shuffle + Linear)
# ============================================================

class Connector(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = SCALE_FACTOR
        self.proj = nn.Linear(CONNECTOR_INPUT_DIM, CONNECTOR_OUTPUT_DIM, bias=False)

    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq ** 0.5)
        sf = self.scale_factor
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / sf), embed_dim * sf)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / sf), int(height / sf), embed_dim * (sf ** 2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (sf ** 2)), embed_dim * (sf ** 2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states)
        image_hidden_states = self.proj(image_hidden_states)
        return image_hidden_states


# ============================================================
# RMSNorm (for LLama-based layers)
# ============================================================

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


# ============================================================
# RoPE
# ============================================================

def apply_rope(x, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
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


# ============================================================
# LLama Decoder Layer (used for both VLM and Expert)
# ============================================================

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_eps):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = LlamaAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_eps)

    def forward(self, hidden_states):
        # This is not used directly - the SmolVLMWithExpert forward handles the layers
        raise NotImplementedError("Use SmolVLMWithExpert.forward()")


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


# ============================================================
# LLama Text Model
# ============================================================

class LlamaTextModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(TEXT_VOCAB_SIZE, TEXT_HIDDEN_SIZE)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                TEXT_HIDDEN_SIZE, TEXT_NUM_HEADS, TEXT_NUM_KV_HEADS,
                TEXT_HEAD_DIM, TEXT_INTERMEDIATE_SIZE, TEXT_RMS_NORM_EPS
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(TEXT_HIDDEN_SIZE, eps=TEXT_RMS_NORM_EPS)

    def get_input_embeddings(self):
        return self.embed_tokens


class ExpertModel(nn.Module):
    """Action expert model (smaller Llama)."""
    def __init__(self):
        super().__init__()
        self.embed_tokens = None  # Removed as in original
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                EXPERT_HIDDEN_SIZE, EXPERT_NUM_HEADS, EXPERT_NUM_KV_HEADS,
                TEXT_HEAD_DIM, EXPERT_INTERMEDIATE_SIZE, TEXT_RMS_NORM_EPS
            ) for _ in range(NUM_VLM_LAYERS)
        ])
        self.norm = RMSNorm(EXPERT_HIDDEN_SIZE, eps=TEXT_RMS_NORM_EPS)

    def _init_cross_attn_kv(self):
        """Replace KV projections for cross-attention layers."""
        vlm_kv_dim = TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM  # 320
        expert_kv_dim = EXPERT_NUM_KV_HEADS * TEXT_HEAD_DIM  # 320
        for layer_idx in range(len(self.layers)):
            if SELF_ATTN_EVERY_N_LAYERS > 0 and layer_idx % SELF_ATTN_EVERY_N_LAYERS == 0:
                continue
            # Cross-attention layers: KV input comes from VLM (dimension = vlm_kv_dim)
            self.layers[layer_idx].self_attn.k_proj = nn.Linear(
                vlm_kv_dim, expert_kv_dim, bias=False
            )
            self.layers[layer_idx].self_attn.v_proj = nn.Linear(
                vlm_kv_dim, expert_kv_dim, bias=False
            )


# ============================================================
# Eager Attention Interface
# ============================================================

def eager_attention_forward(attention_mask, batch_size, head_dim, query_states, key_states, value_states,
                            num_attention_heads=TEXT_NUM_HEADS, num_key_value_heads=TEXT_NUM_KV_HEADS):
    """Eager attention with GQA expansion."""
    num_key_value_groups = num_attention_heads // num_key_value_heads
    sequence_length = key_states.shape[1]

    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    key_states = key_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    value_states = value_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    query_states = query_states.to(dtype=torch.float32)
    key_states = key_states.to(dtype=torch.float32)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

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


# ============================================================
# SmolVLM With Expert
# ============================================================

class SmolVLMWithExpert(nn.Module):
    """Combined VLM + Expert model with cross-attention."""

    def __init__(self):
        super().__init__()
        # Vision encoder
        self.vision_encoder = VisionEncoder()
        # Connector
        self.connector = Connector()
        # VLM text model (16 layers)
        self.text_model = LlamaTextModel(NUM_VLM_LAYERS)
        # Expert model (16 layers with cross-attn on odd layers)
        self.lm_expert = ExpertModel()
        self.lm_expert._init_cross_attn_kv()

        self.num_vlm_layers = NUM_VLM_LAYERS
        self.num_expert_layers = NUM_VLM_LAYERS
        self.num_attention_heads = TEXT_NUM_HEADS
        self.num_key_value_heads = TEXT_NUM_KV_HEADS

    def embed_image(self, image):
        image_hidden_states = self.vision_encoder(image)
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens):
        return self.text_model.get_input_embeddings()(tokens)

    def get_model_layers(self):
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = self.lm_expert.layers[expert_layer_index]
            vlm_layers.append(self.text_model.layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids,
                           attention_mask, batch_size, head_dim,
                           use_cache=True, fill_kv_cache=True, past_key_values=None):
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

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

        if use_cache and past_key_values is None:
            past_key_values = {}
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        att_output = eager_attention_forward(
            _attention_mask, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids,
                                 attention_mask, batch_size, head_dim,
                                 use_cache=True, fill_kv_cache=True, past_key_values=None):
        att_outputs = []

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id = position_ids[:, :seq_len]
            expert_position_id = position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]
            hidden_states = layer.input_layernorm(inputs_embeds[0])
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = eager_attention_forward(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert cross-attention
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1]:, :expert_key_states.shape[1]:
            ]

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = eager_attention_forward(
                expert_attention_mask, batch_size, head_dim,
                expert_query_states, expert_key_states, expert_value_states
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        return att_outputs, past_key_values

    def forward(self, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, fill_kv_cache=None):
        models = [self.text_model, self.lm_expert]
        model_layers = self.get_model_layers()

        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = self.num_vlm_layers
        head_dim = TEXT_HEAD_DIM

        for layer_idx in range(num_layers):
            if (fill_kv_cache
                or "cross" not in ATTENTION_MODE
                or (SELF_ATTN_EVERY_N_LAYERS > 0 and layer_idx % SELF_ATTN_EVERY_N_LAYERS == 0)):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers, inputs_embeds, layer_idx, position_ids,
                    attention_mask, batch_size, head_dim,
                    use_cache=use_cache, fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers, inputs_embeds, layer_idx, position_ids,
                    attention_mask, batch_size, head_dim,
                    use_cache=use_cache, fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )

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

        # Final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values


# ============================================================
# Helper Functions
# ============================================================

def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = torch.float64 if device != "mps" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def pad_tensor(tensor, max_len, pad_value=0):
    b, d = tensor.shape[:2]
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor
    return padded_tensor


# ============================================================
# VLAFlowMatching (Main Model)
# ============================================================

class Model(nn.Module):
    """
    SmolVLA VLAFlowMatching model.

    Self-contained wrapper with all HuggingFace dependencies inlined.
    Config: SmolVLM2-500M-Video-Instruct with 16 VLM layers, cross-attention expert.
    """

    def __init__(self):
        super().__init__()
        self.vlm_with_expert = SmolVLMWithExpert()

        # Projections
        self.state_proj = nn.Linear(MAX_STATE_DIM, TEXT_HIDDEN_SIZE)
        self.action_in_proj = nn.Linear(MAX_ACTION_DIM, EXPERT_HIDDEN_SIZE)
        self.action_out_proj = nn.Linear(EXPERT_HIDDEN_SIZE, MAX_ACTION_DIM)
        self.action_time_mlp_in = nn.Linear(EXPERT_HIDDEN_SIZE * 2, EXPERT_HIDDEN_SIZE)
        self.action_time_mlp_out = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_HIDDEN_SIZE)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks):
            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim ** 0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if PREFIX_LENGTH > 0 and seq_len < PREFIX_LENGTH:
            embs = pad_tensor(embs, PREFIX_LENGTH, pad_value=0)
            pad_masks = pad_tensor(pad_masks, PREFIX_LENGTH, pad_value=0)
            att_masks = pad_tensor(att_masks, PREFIX_LENGTH, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep, EXPERT_HIDDEN_SIZE, MIN_PERIOD, MAX_PERIOD, device=device
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)
        att_masks += [1] * CHUNK_SIZE

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        """Full training forward pass. Returns per-element losses [B, chunk_size, action_dim]."""
        if noise is None:
            noise = torch.normal(mean=0.0, std=1.0, size=actions.shape,
                                 dtype=torch.float32, device=actions.device)
        if time is None:
            beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
            time = beta_dist.sample((actions.shape[0],)).to(device=actions.device, dtype=torch.float32)
            time = time * 0.999 + 0.001

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        suffix_out = suffix_out[:, -CHUNK_SIZE:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses


# ============================================================
# Interface Functions
# ============================================================

def get_inputs():
    """Generate test inputs matching VLAFlowMatching.forward() signature."""
    B = BATCH_SIZE
    images = [torch.randn(B, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)]
    img_masks = [torch.ones(B, dtype=torch.bool)]
    lang_tokens = torch.randint(0, TEXT_VOCAB_SIZE, (B, LANG_SEQ_LEN))
    lang_masks = torch.ones(B, LANG_SEQ_LEN, dtype=torch.bool)
    state = torch.randn(B, MAX_STATE_DIM)
    actions = torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM)
    noise = torch.randn(B, CHUNK_SIZE, MAX_ACTION_DIM)
    time = torch.rand(B) * 0.999 + 0.001
    return [images, img_masks, lang_tokens, lang_masks, state, actions, noise, time]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(BATCH_SIZE, CHUNK_SIZE, MAX_ACTION_DIM)]


def run_tests():
    """Verify this component executes correctly."""
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape] if isinstance(output, torch.Tensor) else [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Output {i} shape mismatch: got {actual}, expected {expected}"
            print(f"Input shapes: images={[img.shape for img in inputs[0]]}, "
                  f"lang_tokens={inputs[2].shape}, state={inputs[4].shape}, "
                  f"actions={inputs[5].shape}")
            print(f"Output shape: {actual_shapes}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
