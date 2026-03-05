"""
Step 1 Refactored: VLAFlowMatching decomposed into Layer-level children.

Children:
  - vision_encoder: SigLIP vision encoder + pixel shuffle connector
  - action_time_mlp: Action-time embedding MLP
  - vlm_expert_transformer: 16-layer VLM+Expert transformer
  - text_embedding: nn.Embedding for language tokens
  - state_proj: nn.Linear for state projection
  - action_out_proj: nn.Linear for action output projection

forward() only does data plumbing: concatenation, mask building, arithmetic
for flow matching (x_t, u_t), and calling children.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Import child modules
_children_dir = str(Path(__file__).resolve().parent / "children")
sys.path.insert(0, _children_dir)

import vision_encoder as vision_encoder_mod
import action_time_mlp as action_time_mlp_mod
import vlm_expert_transformer as vlm_expert_transformer_mod

# Constants
TEXT_HIDDEN_SIZE = 960
TEXT_VOCAB_SIZE = 49280
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
EXPERT_HIDDEN_SIZE = 720
CHUNK_SIZE = 50
BATCH_SIZE = 1
LANG_SEQ_LEN = 48
VISION_IMAGE_SIZE = 512
NUM_IMAGE_TOKENS = 64
PREFIX_LENGTH = -1


def make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


class RefactoredModel(nn.Module):
    """VLAFlowMatching decomposed into layer-level children."""

    def __init__(self):
        super().__init__()
        self.vision_encoder = vision_encoder_mod.Model()
        self.action_time_mlp = action_time_mlp_mod.Model()
        self.vlm_expert_transformer = vlm_expert_transformer_mod.Model()
        self.text_embedding = nn.Embedding(TEXT_VOCAB_SIZE, TEXT_HIDDEN_SIZE)
        self.state_proj = nn.Linear(MAX_STATE_DIM, TEXT_HIDDEN_SIZE)
        self.action_out_proj = nn.Linear(EXPERT_HIDDEN_SIZE, MAX_ACTION_DIM)

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        # Flow matching: compute noisy actions and target
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

        # === Build prefix embeddings ===
        embs = []
        pad_masks_list = []
        att_masks_list = []

        # Image embeddings via vision_encoder child
        for img, img_mask in zip(images, img_masks):
            img_emb = self.vision_encoder(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * (img_emb_dim ** 0.5)
            bsize, num_img_embs = img_emb.shape[:2]
            _img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks_list.append(_img_mask)
            att_masks_list += [0] * num_img_embs

        # Language embeddings via text_embedding child
        lang_emb = self.text_embedding(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks_list.append(lang_masks)
        att_masks_list += [0] * lang_emb.shape[1]

        # State embedding via state_proj child
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks_list.append(state_mask)
        att_masks_list += [1] * states_seq_len

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks_list, dim=1)
        prefix_att_masks = torch.tensor(att_masks_list, dtype=torch.bool, device=prefix_pad_masks.device)
        prefix_att_masks = prefix_att_masks[None, :].expand(bsize, -1)

        # === Build suffix embeddings via action_time_mlp child ===
        suffix_embs = self.action_time_mlp(x_t, time)
        bsize_s, action_time_dim = suffix_embs.shape[:2]
        suffix_pad_masks = torch.ones(bsize_s, action_time_dim, dtype=torch.bool, device=device)
        suffix_att_masks = torch.ones(bsize_s, CHUNK_SIZE, dtype=suffix_embs.dtype, device=suffix_embs.device)

        # === Build attention masks and position IDs ===
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # === Forward through VLM+Expert transformer child ===
        suffix_out = self.vlm_expert_transformer(prefix_embs, suffix_embs, att_2d_masks, position_ids)

        # === Action output via action_out_proj child ===
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        # MSE loss (element-wise, no reduction) - using allowed arithmetic
        losses = (u_t - v_t) ** 2
        return losses


def get_inputs():
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
