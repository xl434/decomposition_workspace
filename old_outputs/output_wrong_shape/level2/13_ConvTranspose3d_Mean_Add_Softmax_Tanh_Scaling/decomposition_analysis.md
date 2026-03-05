# Decomposition Analysis: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling

## Classification
- **Level**: 2 (Layer)
- **Operations**: 6 (ConvTranspose3d, Mean, BiasAdd, Softmax, Tanh, Scaling)

## Data Flow
```
Input: [2, 2, 4, 4, 4]
  -> ConvTranspose3d(2, 4, 3, stride=1, padding=1): [2, 4, 4, 4, 4]
  -> Mean(dim=2, keepdim=True):                      [2, 4, 1, 4, 4]
  -> BiasAdd (bias shape [1,4,1,1,1]):                [2, 4, 1, 4, 4]
  -> Softmax(dim=1):                                  [2, 4, 1, 4, 4]
  -> Tanh:                                            [2, 4, 1, 4, 4]
  -> Scale (*2.0):                                    [2, 4, 1, 4, 4]
Output: [2, 4, 1, 4, 4]
```

## Hierarchy

### Level 2 (Layer): conv_mean_softmax_tanh_scale
Full model combining both fusions. File: `level_2_layer/conv_mean_softmax_tanh_scale.py`

### Level 1 (Fusion):
1. **conv_mean_bias**: ConvTranspose3d -> Mean -> BiasAdd (spatial computation group)
   - File: `level_1_fusion/conv_mean_bias.py`
2. **softmax_tanh_scale**: Softmax -> Tanh -> Scale (activation pipeline)
   - File: `level_1_fusion/softmax_tanh_scale.py`

### Level 0 (Kernel):
1. **conv_transpose3d**: `level_0_kernel/conv_transpose3d.py`
2. **mean_pool**: `level_0_kernel/mean_pool.py`
3. **bias_add**: `level_0_kernel/bias_add.py`
4. **softmax**: `level_0_kernel/softmax.py`
5. **tanh**: `level_0_kernel/tanh.py`
6. **scaling**: `level_0_kernel/scaling.py`

## Fusion Rationale
- **Fusion 1 (conv_mean_bias)**: Groups the spatial computation operations -- the transposed convolution produces a 3D feature map, the mean reduces the depth dimension, and the bias is added per-channel. These share spatial data locality.
- **Fusion 2 (softmax_tanh_scale)**: Groups the pointwise activation pipeline -- softmax normalizes across channels, tanh applies a nonlinearity, and scaling adjusts the magnitude. These are all element-wise or channel-wise operations that can be fused for memory bandwidth savings.

## Test Configuration
- batch_size=2, in_channels=2, out_channels=4
- depth=4, height=4, width=4
- kernel_size=3, stride=1, padding=1
- scaling_factor=2.0
