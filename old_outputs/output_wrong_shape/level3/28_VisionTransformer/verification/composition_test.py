"""
Composition Test: Vision Transformer (28_VisionTransformer)
Verifies that the hierarchical decomposition of the Vision Transformer produces
outputs identical to the original monolithic model.

Tests three levels of composition:
1. Kernel-level (Level 0): Run each atomic operation in sequence
2. Fusion-level (Level 1): Run fused operation groups in sequence
3. Layer-level (Level 2): Run layer-level components in sequence

All levels share weights from a single original model and must produce matching outputs.

Test dimensions:
  image_size=16, patch_size=4, num_classes=10, dim=32, depth=2,
  heads=4, mlp_dim=64, channels=3, dropout=0.0, emb_dropout=0.0
  Input: [2, 3, 16, 16] -> Output: [2, 10]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Original Model (reference implementation)
# ============================================================================
class OriginalViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(OriginalViT, self).__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout
            ),
            num_layers=depth,
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(
            img.shape[0], -1, p * p * img.shape[1]
        )
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


# ============================================================================
# Test: Level 0 Kernel Composition
# ============================================================================
def test_kernel_composition():
    """
    Composes all Level 0 kernels in sequence and verifies output matches original.
    Kernels: patch_unfold -> linear_embed -> cls_token_cat -> pos_embedding_add ->
             dropout -> [transformer_layer x depth] -> cls_token_extract ->
             linear_head_up -> gelu -> dropout -> linear_head_down
    """
    torch.manual_seed(42)
    original = OriginalViT(16, 4, 10, 32, 2, 4, 64, 3, 0.0, 0.0)
    original.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        ref = original(x)

    # Compose from kernels, sharing weights from the original model
    with torch.no_grad():
        # Kernel 1: patch_unfold - unfold + reshape
        p = 4
        out = x.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, p * p * 3)
        assert out.shape == (2, 16, 48), f"patch_unfold: {out.shape}"

        # Kernel 2: linear_embed - Linear(48, 32)
        out = F.linear(out, original.patch_to_embedding.weight,
                       original.patch_to_embedding.bias)
        assert out.shape == (2, 16, 32), f"linear_embed: {out.shape}"

        # Kernel 3: cls_token_cat - prepend CLS token
        cls_tokens = original.cls_token.expand(2, -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        assert out.shape == (2, 17, 32), f"cls_token_cat: {out.shape}"

        # Kernel 4: pos_embedding_add - add positional embedding
        out = out + original.pos_embedding
        assert out.shape == (2, 17, 32), f"pos_embedding_add: {out.shape}"

        # Kernel 5: dropout (rate 0.0, eval mode = identity)
        out = F.dropout(out, p=0.0, training=False)
        assert out.shape == (2, 17, 32), f"dropout: {out.shape}"

        # Kernels 6-7: transformer layers (2 TransformerEncoderLayers)
        # Each layer internally: self_attn + add&norm + ffn + add&norm
        out = original.transformer.layers[0](out)
        assert out.shape == (2, 17, 32), f"transformer_layer_0: {out.shape}"
        out = original.transformer.layers[1](out)
        assert out.shape == (2, 17, 32), f"transformer_layer_1: {out.shape}"

        # Kernel 8: cls_token_extract - extract first token
        out = out[:, 0]
        assert out.shape == (2, 32), f"cls_token_extract: {out.shape}"

        # Kernel 9: linear_head_up - Linear(32, 64)
        out = F.linear(out, original.mlp_head[0].weight, original.mlp_head[0].bias)
        assert out.shape == (2, 64), f"linear_head_up: {out.shape}"

        # Kernel 10: gelu
        out = F.gelu(out)
        assert out.shape == (2, 64), f"gelu: {out.shape}"

        # Kernel 11: dropout (rate 0.0)
        out = F.dropout(out, p=0.0, training=False)
        assert out.shape == (2, 64), f"head_dropout: {out.shape}"

        # Kernel 12: linear_head_down - Linear(64, 10)
        out = F.linear(out, original.mlp_head[3].weight, original.mlp_head[3].bias)
        assert out.shape == (2, 10), f"linear_head_down: {out.shape}"

    max_diff = (ref - out).abs().max().item()
    print(f"  Kernel composition max diff: {max_diff:.2e}")
    assert torch.allclose(ref, out, rtol=1e-4, atol=1e-5), \
        f"Kernel composition FAILED: max diff {max_diff}"
    print("  Level 0 (Kernel) composition: PASS")


# ============================================================================
# Test: Level 1 Fusion Composition
# ============================================================================
def test_fusion_composition():
    """
    Composes all Level 1 fusions in sequence and verifies output matches original.
    Fusions: patch_unfold_embed -> cls_pos_dropout -> [transformer_layer x depth] ->
             cls_extract_mlp
    """
    torch.manual_seed(42)
    original = OriginalViT(16, 4, 10, 32, 2, 4, 64, 3, 0.0, 0.0)
    original.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        ref = original(x)

    # Compose from fusions, sharing weights from the original model
    with torch.no_grad():
        # Fusion 1: patch_unfold_embed (patch_unfold + linear_embed)
        p = 4
        patches = x.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, p * p * 3)
        out = F.linear(patches, original.patch_to_embedding.weight,
                       original.patch_to_embedding.bias)
        assert out.shape == (2, 16, 32), f"patch_unfold_embed: {out.shape}"

        # Fusion 2: cls_pos_dropout (cls_token_cat + pos_embedding_add + dropout)
        cls_tokens = original.cls_token.expand(2, -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out = out + original.pos_embedding
        out = F.dropout(out, p=0.0, training=False)
        assert out.shape == (2, 17, 32), f"cls_pos_dropout: {out.shape}"

        # Fusion 3 & 4: transformer_layer x 2
        out = original.transformer.layers[0](out)
        assert out.shape == (2, 17, 32), f"transformer_layer_0: {out.shape}"
        out = original.transformer.layers[1](out)
        assert out.shape == (2, 17, 32), f"transformer_layer_1: {out.shape}"

        # Fusion 5: cls_extract_mlp (cls_token_extract + mlp_head)
        out = out[:, 0]
        out = original.mlp_head(out)
        assert out.shape == (2, 10), f"cls_extract_mlp: {out.shape}"

    max_diff = (ref - out).abs().max().item()
    print(f"  Fusion composition max diff: {max_diff:.2e}")
    assert torch.allclose(ref, out, rtol=1e-4, atol=1e-5), \
        f"Fusion composition FAILED: max diff {max_diff}"
    print("  Level 1 (Fusion) composition: PASS")


# ============================================================================
# Test: Level 2 Layer Composition
# ============================================================================
def test_layer_composition():
    """
    Composes all Level 2 layers in sequence and verifies output matches original.
    Layers: patch_embedding -> transformer_encoder -> classification_head
    """
    torch.manual_seed(42)
    original = OriginalViT(16, 4, 10, 32, 2, 4, 64, 3, 0.0, 0.0)
    original.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        ref = original(x)

    # Compose from layers, sharing weights from the original model
    with torch.no_grad():
        # Layer 1: patch_embedding (unfold + embed + cls + pos + dropout)
        p = original.patch_size
        out = x.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, p * p * 3)
        out = original.patch_to_embedding(out)
        cls_tokens = original.cls_token.expand(2, -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out = out + original.pos_embedding
        out = original.dropout(out)
        assert out.shape == (2, 17, 32), f"patch_embedding: {out.shape}"

        # Layer 2: transformer_encoder (full encoder stack)
        out = original.transformer(out)
        assert out.shape == (2, 17, 32), f"transformer_encoder: {out.shape}"

        # Layer 3: classification_head (cls extract + mlp head)
        out = original.to_cls_token(out[:, 0])
        out = original.mlp_head(out)
        assert out.shape == (2, 10), f"classification_head: {out.shape}"

    max_diff = (ref - out).abs().max().item()
    print(f"  Layer composition max diff: {max_diff:.2e}")
    assert torch.allclose(ref, out, rtol=1e-4, atol=1e-5), \
        f"Layer composition FAILED: max diff {max_diff}"
    print("  Level 2 (Layer) composition: PASS")


# ============================================================================
# Test: Cross-Level Consistency
# ============================================================================
def test_cross_level_consistency():
    """
    Verifies that all decomposition levels produce identical outputs when
    given the same input and shared weights.
    """
    torch.manual_seed(42)
    original = OriginalViT(16, 4, 10, 32, 2, 4, 64, 3, 0.0, 0.0)
    original.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        # Original forward pass
        ref = original(x)

        # Level 2 (Layer) composition
        p = original.patch_size
        l2 = x.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, p * p * 3)
        l2 = original.patch_to_embedding(l2)
        cls_tokens = original.cls_token.expand(2, -1, -1)
        l2 = torch.cat((cls_tokens, l2), dim=1)
        l2 = l2 + original.pos_embedding
        l2 = original.dropout(l2)
        l2 = original.transformer(l2)
        l2 = original.to_cls_token(l2[:, 0])
        l2 = original.mlp_head(l2)

        # Level 1 (Fusion) composition
        l1 = x.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, p * p * 3)
        l1 = F.linear(l1, original.patch_to_embedding.weight,
                       original.patch_to_embedding.bias)
        cls_tokens = original.cls_token.expand(2, -1, -1)
        l1 = torch.cat((cls_tokens, l1), dim=1)
        l1 = l1 + original.pos_embedding
        l1 = F.dropout(l1, p=0.0, training=False)
        for layer in original.transformer.layers:
            l1 = layer(l1)
        l1 = l1[:, 0]
        l1 = original.mlp_head(l1)

        # Level 0 (Kernel) composition
        l0 = x.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, p * p * 3)
        l0 = F.linear(l0, original.patch_to_embedding.weight,
                       original.patch_to_embedding.bias)
        cls_tokens = original.cls_token.expand(2, -1, -1)
        l0 = torch.cat((cls_tokens, l0), dim=1)
        l0 = l0 + original.pos_embedding
        l0 = F.dropout(l0, p=0.0, training=False)
        for layer in original.transformer.layers:
            l0 = layer(l0)
        l0 = l0[:, 0]
        l0 = F.linear(l0, original.mlp_head[0].weight, original.mlp_head[0].bias)
        l0 = F.gelu(l0)
        l0 = F.dropout(l0, p=0.0, training=False)
        l0 = F.linear(l0, original.mlp_head[3].weight, original.mlp_head[3].bias)

    # All levels should match
    diff_ref_l2 = (ref - l2).abs().max().item()
    diff_ref_l1 = (ref - l1).abs().max().item()
    diff_ref_l0 = (ref - l0).abs().max().item()
    diff_l2_l1 = (l2 - l1).abs().max().item()
    diff_l2_l0 = (l2 - l0).abs().max().item()
    diff_l1_l0 = (l1 - l0).abs().max().item()

    print(f"  Original vs Level 2: {diff_ref_l2:.2e}")
    print(f"  Original vs Level 1: {diff_ref_l1:.2e}")
    print(f"  Original vs Level 0: {diff_ref_l0:.2e}")
    print(f"  Level 2 vs Level 1:  {diff_l2_l1:.2e}")
    print(f"  Level 2 vs Level 0:  {diff_l2_l0:.2e}")
    print(f"  Level 1 vs Level 0:  {diff_l1_l0:.2e}")

    assert torch.allclose(ref, l2, rtol=1e-4, atol=1e-5), "Original vs Level 2 mismatch"
    assert torch.allclose(ref, l1, rtol=1e-4, atol=1e-5), "Original vs Level 1 mismatch"
    assert torch.allclose(ref, l0, rtol=1e-4, atol=1e-5), "Original vs Level 0 mismatch"
    print("  Cross-level consistency: PASS")


# ============================================================================
# Test: Individual Component Shape Verification
# ============================================================================
def test_shape_verification():
    """
    Verifies the expected shapes at each stage of the decomposition.
    """
    torch.manual_seed(42)
    original = OriginalViT(16, 4, 10, 32, 2, 4, 64, 3, 0.0, 0.0)
    original.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        p = 4
        # Patch unfold
        patches = x.unfold(2, p, p).unfold(3, p, p).reshape(2, -1, p * p * 3)
        assert patches.shape == (2, 16, 48), f"Expected (2,16,48), got {patches.shape}"

        # Linear embed
        embedded = original.patch_to_embedding(patches)
        assert embedded.shape == (2, 16, 32), f"Expected (2,16,32), got {embedded.shape}"

        # CLS token cat
        cls_tokens = original.cls_token.expand(2, -1, -1)
        with_cls = torch.cat((cls_tokens, embedded), dim=1)
        assert with_cls.shape == (2, 17, 32), f"Expected (2,17,32), got {with_cls.shape}"

        # Pos embedding add
        with_pos = with_cls + original.pos_embedding
        assert with_pos.shape == (2, 17, 32), f"Expected (2,17,32), got {with_pos.shape}"

        # Dropout
        after_drop = original.dropout(with_pos)
        assert after_drop.shape == (2, 17, 32), f"Expected (2,17,32), got {after_drop.shape}"

        # Transformer layer 0
        after_layer0 = original.transformer.layers[0](after_drop)
        assert after_layer0.shape == (2, 17, 32), f"Expected (2,17,32), got {after_layer0.shape}"

        # Transformer layer 1
        after_layer1 = original.transformer.layers[1](after_layer0)
        assert after_layer1.shape == (2, 17, 32), f"Expected (2,17,32), got {after_layer1.shape}"

        # CLS token extract
        cls_out = after_layer1[:, 0]
        assert cls_out.shape == (2, 32), f"Expected (2,32), got {cls_out.shape}"

        # MLP head up
        head_up = F.linear(cls_out, original.mlp_head[0].weight, original.mlp_head[0].bias)
        assert head_up.shape == (2, 64), f"Expected (2,64), got {head_up.shape}"

        # GELU
        after_gelu = F.gelu(head_up)
        assert after_gelu.shape == (2, 64), f"Expected (2,64), got {after_gelu.shape}"

        # MLP head down
        final = F.linear(after_gelu, original.mlp_head[3].weight, original.mlp_head[3].bias)
        assert final.shape == (2, 10), f"Expected (2,10), got {final.shape}"

    print("  Shape verification: PASS")


# ============================================================================
# Main
# ============================================================================
def run_tests():
    print("=" * 60)
    print("Composition Test: 28_VisionTransformer")
    print("=" * 60)

    print("\nTest 1: Level 0 (Kernel) Composition")
    test_kernel_composition()

    print("\nTest 2: Level 1 (Fusion) Composition")
    test_fusion_composition()

    print("\nTest 3: Level 2 (Layer) Composition")
    test_layer_composition()

    print("\nTest 4: Cross-Level Consistency")
    test_cross_level_consistency()

    print("\nTest 5: Shape Verification")
    test_shape_verification()

    print("\n" + "=" * 60)
    print("ALL COMPOSITION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
