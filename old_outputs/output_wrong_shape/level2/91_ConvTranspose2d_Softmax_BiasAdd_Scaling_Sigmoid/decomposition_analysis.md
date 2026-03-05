# Decomposition Analysis: 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid

## Classification
- **Level**: 1 (Fusion)
- **Operations**: 5 (ConvTranspose2d, Softmax, BiasAdd, Scaling, Sigmoid)
- **Pattern**: Linear sequential pipeline with no branching

## Architecture Overview
This model implements a transposed convolution followed by a sequence of pointwise and channel-wise operations. The data flows strictly sequentially through all five operations.

## Data Flow
```
Input [2, 3, 4, 4]
  |
  v
ConvTranspose2d(3->8, k=4, s=2, p=1, op=1) --> [2, 8, 9, 9]
  |
  v
Softmax(dim=1)                                --> [2, 8, 9, 9]
  |
  v
BiasAdd (bias: [8,1,1])                       --> [2, 8, 9, 9]
  |
  v
Scaling (* 2.0)                               --> [2, 8, 9, 9]
  |
  v
Sigmoid                                       --> [2, 8, 9, 9]
```

## Output Size Calculation
ConvTranspose2d output formula: `H' = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding`
- H' = (4 - 1) * 2 - 2 * 1 + 4 + 1 = 6 - 2 + 4 + 1 = 9
- W' = 9 (same calculation)

## Decomposition Strategy
All five operations are decomposed into individual Level 0 kernels. Since this is a Level 1 fusion (single sequential chain), the fusion file simply chains all kernels together.

### Level 0 Kernels
| Kernel | Operation | Learnable Params | Notes |
|--------|-----------|-----------------|-------|
| conv_transpose2d | nn.ConvTranspose2d | weight [3,8,4,4], bias [8] | Main compute-heavy operation |
| softmax | torch.softmax(dim=1) | none | Channel-wise normalization |
| bias_add | x + bias | bias [8,1,1] | Broadcasts across spatial dims |
| scaling | x * 2.0 | none | Constant multiplication |
| sigmoid | torch.sigmoid | none | Elementwise activation |

### Fusion Opportunities
- **Softmax + BiasAdd + Scaling + Sigmoid**: These four pointwise/channel-wise operations after the convolution are strong candidates for kernel fusion since they all operate elementwise (or along a single dimension) on the same tensor shape [2, 8, 9, 9].
- **Scaling + Sigmoid**: Can be fused into a single scaled sigmoid: `sigmoid(2.0 * x)` is equivalent to `sigmoid(x * 2.0)` which avoids an intermediate tensor allocation.

## Test Dimensions
Small test dimensions used (batch_size=2) compared to original (batch_size=128) to enable fast verification:
- batch_size: 2 (original: 128)
- in_channels: 3 (original: 64)
- out_channels: 8 (original: 128)
- height, width: 4, 4 (original: 64, 64)
- kernel_size: 4 (unchanged)
- stride: 2 (unchanged)
- padding: 1 (unchanged)
- output_padding: 1 (unchanged)
- scaling_factor: 2.0 (unchanged)
