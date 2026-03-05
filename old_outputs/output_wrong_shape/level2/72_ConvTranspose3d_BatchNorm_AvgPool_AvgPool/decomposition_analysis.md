# Decomposition Analysis: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool

## Overview
- **Model**: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool
- **Classification**: Level 1 (Fusion)
- **Total Operations**: 4
- **Decomposition Depth**: 1 level (Level 0 kernels composed into Level 1 fusion)

## Original Model
```python
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        return x
```

## Production Dimensions
- batch_size=64, in_channels=3, out_channels=16
- depth=32, height=32, width=32
- kernel_size=3, stride=2, padding=1
- bias_shape=(16, 1, 1, 1)

## Test Dimensions
- batch_size=2, in_channels=2, out_channels=4
- depth=8, height=8, width=8
- kernel_size=3, stride=2, padding=1
- bias_shape=(4, 1, 1, 1)

## Data Flow (Test Dimensions)
```
Input: [2, 2, 8, 8, 8]
  -> ConvTranspose3d(2, 4, k=3, s=2, p=1)  -> [2, 4, 15, 15, 15]
  -> BatchNorm3d(4)                          -> [2, 4, 15, 15, 15]
  -> AvgPool3d(kernel_size=2)                -> [2, 4, 7, 7, 7]
  -> AvgPool3d(kernel_size=2)                -> [2, 4, 3, 3, 3]
```

### Dimension Calculation
- ConvTranspose3d output: (input - 1) * stride - 2 * padding + kernel = (8-1)*2 - 2*1 + 3 = 15
- AvgPool3d(2) on 15: floor(15/2) = 7
- AvgPool3d(2) on 7: floor(7/2) = 3

## Decomposition Tree

```
72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool (Level 1 - Fusion)
├── conv_transpose3d (Level 0 - Kernel) : nn.ConvTranspose3d
├── batch_norm3d     (Level 0 - Kernel) : nn.BatchNorm3d
├── avg_pool3d_1     (Level 0 - Kernel) : nn.AvgPool3d
└── avg_pool3d_2     (Level 0 - Kernel) : nn.AvgPool3d
```

## Component Summary

| Component       | Level | Type   | Input Shape          | Output Shape         | Parameters |
|-----------------|-------|--------|----------------------|----------------------|------------|
| conv_transpose3d| 0     | kernel | [2,2,8,8,8]         | [2,4,15,15,15]      | weight[2,4,3,3,3], bias[4] |
| batch_norm3d    | 0     | kernel | [2,4,15,15,15]      | [2,4,15,15,15]      | weight[4], bias[4], running_mean[4], running_var[4] |
| avg_pool3d_1    | 0     | kernel | [2,4,15,15,15]      | [2,4,7,7,7]         | none |
| avg_pool3d_2    | 0     | kernel | [2,4,7,7,7]         | [2,4,3,3,3]         | none |
| conv_bn_pool_pool| 1    | fusion | [2,2,8,8,8]         | [2,4,3,3,3]         | all of above |

## Notes
- The ConvTranspose3d upsamples the spatial dimensions from 8 to 15.
- BatchNorm3d normalizes across the channel dimension and preserves spatial shape.
- Two successive AvgPool3d layers with kernel_size=2 progressively downsample: 15 -> 7 -> 3.
- The pooling layers are parameter-free and deterministic.
- The bias_shape parameter in the original constructor is accepted but not used (ConvTranspose3d has its own bias).
- The sequential chain has no branching, making decomposition straightforward.
