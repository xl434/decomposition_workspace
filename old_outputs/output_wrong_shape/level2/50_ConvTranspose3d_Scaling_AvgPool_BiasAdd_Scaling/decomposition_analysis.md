# Decomposition Analysis: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling

## Overview

This model implements a sequential pipeline of 5 operations for 3D volumetric data processing:
1. 3D transposed convolution (volumetric upsampling)
2. Scalar scaling (multiply by learnable parameter, init 0.5)
3. 3D average pooling (spatial downsampling)
4. Bias addition (learnable per-channel bias with broadcasting)
5. Scalar scaling (multiply by learnable parameter, init 1.0)

## Classification

- **Level**: 1 (Fusion)
- **Number of operations**: 5
- **Architecture pattern**: Sequential pipeline (no branching)

## Data Flow

```
Input: [2, 2, 4, 4, 4]
  |
  v
ConvTranspose3d(2, 4, 3, stride=2, padding=1)
  -> [2, 4, 7, 7, 7]
  |
  v
Scale1 (multiply by 0.5)
  -> [2, 4, 7, 7, 7]
  |
  v
AvgPool3d(kernel_size=2)
  -> [2, 4, 3, 3, 3]
  |
  v
BiasAdd (bias shape [4,1,1,1], broadcasts over [B,C,D,H,W])
  -> [2, 4, 3, 3, 3]
  |
  v
Scale2 (multiply by 1.0)
  -> [2, 4, 3, 3, 3]
```

## Decomposition Tree

```
level_1_fusion/conv_scale_pool_bias_scale.py (Level 1 - Fusion)
  +-- level_0_kernel/conv_transpose3d.py (Level 0 - Kernel)
  +-- level_0_kernel/scale1.py (Level 0 - Kernel)
  +-- level_0_kernel/avg_pool3d.py (Level 0 - Kernel)
  +-- level_0_kernel/bias_add.py (Level 0 - Kernel)
  +-- level_0_kernel/scale2.py (Level 0 - Kernel)
```

## Test Parameters (reduced for testing)

| Parameter     | Original | Test         |
|---------------|----------|--------------|
| batch_size    | 128      | 2            |
| in_channels   | 3        | 2            |
| out_channels  | 16       | 4            |
| depth         | 16       | 4            |
| height, width | 32       | 4            |
| kernel_size   | 3        | 3            |
| stride        | 2        | 2            |
| padding       | 1        | 1            |
| scale1        | 0.5      | 0.5          |
| scale2        | 1.0      | 1.0          |
| bias_shape    | (16,1,1,1) | (4,1,1,1) |

## Broadcasting Note

The bias parameter has shape `(out_channels, 1, 1, 1)` which is 4D. When added to a 5D tensor `[B, C, D, H, W]`, PyTorch broadcasting automatically prepends a dimension, effectively treating it as `(1, out_channels, 1, 1, 1)`. This broadcasts correctly across the batch and all spatial dimensions.

## Fusion Opportunities

- **Scale1 + AvgPool3d**: The scalar multiplication can be folded into the pooling operation.
- **BiasAdd + Scale2**: The bias addition and subsequent scaling can be combined into a single affine operation: `x * scale2 + bias * scale2`.
- **Full pipeline**: ConvTranspose3d output can be scaled, pooled, bias-added, and scaled again in a single fused kernel to minimize memory traffic on volumetric data.

## Verification

The composition test (`verification/composition_test.py`) confirms that sequentially executing the 5 individual kernels with shared weights produces output identical to the fused model (within tolerance rtol=1e-4, atol=1e-5).
