# SwinMLP Hierarchical Decomposition Analysis

## Model Overview

SwinMLP is a variant of the Swin Transformer that replaces the self-attention mechanism with a spatial MLP implemented via grouped Conv1d. This eliminates the quadratic complexity of attention while maintaining the hierarchical window-based architecture.

## Architecture Summary

```
Input [2, 3, 32, 32]
  |
  v
PatchEmbed (Conv2d + flatten + transpose + LayerNorm)
  -> [2, 64, 16]
  |
  v
BasicLayer 0 (SwinMLPBlock x1 + PatchMerging)
  -> [2, 16, 32]
  |
  v
BasicLayer 1 (SwinMLPBlock x1 + PatchMerging)
  -> [2, 4, 64]
  |
  v
BasicLayer 2 (SwinMLPBlock x1 + PatchMerging)
  -> [2, 1, 128]
  |
  v
BasicLayer 3 (SwinMLPBlock x1, no PatchMerging)
  -> [2, 1, 128]
  |
  v
LayerNorm + AvgPool + Flatten + Linear
  -> [2, 10]
```

## Test Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| img_size | 32 | Small for fast testing |
| patch_size | 4 | Standard |
| in_chans | 3 | RGB |
| num_classes | 10 | Small classification |
| embed_dim | 16 | Minimal (original: 96) |
| depths | [1,1,1,1] | 1 block per stage (original: [2,2,6,2]) |
| num_heads | [2,4,4,8] | Scaled down |
| window_size | 4 | Fits 8x8 patches |
| mlp_ratio | 2.0 | Reduced from 4.0 |
| batch_size | 2 | Small batch |

## Decomposition Strategy

### Level 3: Full Model
- **swin_mlp.py**: Complete self-contained SwinMLP with all helper classes (Mlp, DropPath, window_partition/reverse, SwinMLPBlock, PatchMerging, BasicLayer, PatchEmbed, Model).

### Level 2: Major Layers (3 files)
- **patch_embed.py**: Image-to-patch embedding via Conv2d projection + normalization.
- **basic_layer_0.py**: Representative BasicLayer with SwinMLPBlock + PatchMerging. Contains the full SwinMLP block implementation.
- **classifier_head.py**: Final classification pipeline (LayerNorm + AvgPool + Linear).

### Level 1: Fused Operations (3 files)
- **conv_proj_norm.py**: Conv2d patch projection fused with LayerNorm.
- **spatial_mlp_block.py**: The KEY novel component - spatial mixing via windowed grouped Conv1d that replaces self-attention.
- **ffn_block.py**: Feed-forward network with residual connection.

### Level 0: Atomic Kernels (10 files)
1. **conv2d_patch.py**: Conv2d(3, 16, 4, stride=4)
2. **flatten_transpose.py**: Spatial-to-sequence reshape
3. **layer_norm.py**: LayerNorm(16)
4. **spatial_conv1d.py**: Conv1d(32, 32, k=1, groups=2) - spatial mixing
5. **linear_ffn_up.py**: Linear(16, 32) - FFN up-projection
6. **gelu.py**: GELU activation
7. **linear_ffn_down.py**: Linear(32, 16) - FFN down-projection
8. **linear_reduction.py**: Linear(64, 32, bias=False) - PatchMerging reduction
9. **adaptive_avg_pool1d.py**: AdaptiveAvgPool1d(1) - global pooling
10. **linear_head.py**: Linear(128, 10) - classification

## Key Novel Component: Spatial MLP

The signature innovation of SwinMLP is replacing self-attention with spatial MLP:

**Standard Swin Transformer:**
```
Q, K, V = linear_qkv(x)       # Project to Q, K, V
attn = softmax(Q @ K^T / sqrt(d)) @ V  # O(n^2) attention
```

**SwinMLP:**
```
x_windows = window_partition(x)           # Partition into local windows
x_heads = reshape(x_windows, groups=num_heads)  # Group by heads
x_mixed = Conv1d(x_heads, groups=num_heads)     # Spatial mixing per head
x = window_reverse(reshape(x_mixed))     # Reconstruct spatial layout
```

The Conv1d with `kernel_size=1` and `groups=num_heads` acts as a learned spatial mixing matrix. Each head independently learns how to mix spatial positions within its window. This is O(n) vs O(n^2) for attention.

## Resolution Adaptation

A critical implementation detail: when the feature map resolution becomes smaller than the window size, the window size is clamped:

```python
if min(input_resolution) <= window_size:
    shift_size = 0
    window_size = min(input_resolution)
```

This happens at layers 2 and 3 in our test configuration:
- Layer 2: resolution (2,2), window_size clamped to 2
- Layer 3: resolution (1,1), window_size clamped to 1

## File Count Summary

| Level | Files | Description |
|-------|-------|-------------|
| Level 3 | 1 | Full model |
| Level 2 | 3 | Major layers |
| Level 1 | 3 | Fused operations |
| Level 0 | 10 | Atomic kernels |
| Verification | 1 | Composition test |
| Metadata | 2 | JSON tree + this analysis |
| **Total** | **20** | |
