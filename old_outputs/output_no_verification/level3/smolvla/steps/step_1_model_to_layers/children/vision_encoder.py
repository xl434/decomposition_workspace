"""
Component: Vision Encoder (SigLIP + Connector)
Abstraction Level: layer
Parent: VLAFlowMatching (root)
Children: [vision_attention, vision_mlp, vision_layernorm, patch_embedding, position_embedding, pixel_shuffle, connector_projection]

Operations: Conv2d patch embedding, position embedding, 12x (LayerNorm + MultiHeadAttention + LayerNorm + MLP),
            post LayerNorm, pixel shuffle, linear projection

Input Shapes:
  - pixel_values: [B, 3, 512, 512] float32

Output Shapes:
  - image_embeddings: [B, 64, 960] float32

Weight Shapes:
  - patch_embedding: Conv2d(3, 768, kernel_size=16, stride=16)
  - position_embedding: Embedding(1024, 768)
  - 12 encoder layers: each with self_attn (q/k/v/out_proj 768x768 + bias) + LayerNorm(768) x2 + MLP(768->3072->768 + bias)
  - post_layernorm: LayerNorm(768)
  - connector.proj: Linear(12288, 960, bias=False)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Config constants
VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_HEAD_DIM = 64
VISION_INTERMEDIATE_SIZE = 3072
VISION_NUM_LAYERS = 12
VISION_IMAGE_SIZE = 512
VISION_PATCH_SIZE = 16
VISION_NUM_CHANNELS = 3
VISION_LAYER_NORM_EPS = 1e-6
VISION_NUM_PATCHES_PER_SIDE = VISION_IMAGE_SIZE // VISION_PATCH_SIZE  # 32
VISION_NUM_PATCHES = VISION_NUM_PATCHES_PER_SIDE ** 2  # 1024
SCALE_FACTOR = 4
CONNECTOR_INPUT_DIM = VISION_HIDDEN_SIZE * (SCALE_FACTOR ** 2)  # 12288
CONNECTOR_OUTPUT_DIM = 960


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
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([VisionEncoderLayer() for _ in range(VISION_NUM_LAYERS)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


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


class Model(nn.Module):
    """Vision Encoder: SigLIP vision transformer + pixel shuffle connector."""
    def __init__(self):
        super().__init__()
        self.embeddings = VisionEmbeddings()
        self.encoder = VisionEncoderBlock()
        self.post_layernorm = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)
        self.connector = Connector()

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        image_embeddings = self.connector(hidden_states)
        return image_embeddings


def get_inputs():
    return [torch.randn(1, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 64, 960)]

def run_tests():
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
            actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Output {i} shape mismatch: got {actual}, expected {expected}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
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
