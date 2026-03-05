# Decomposition Analysis: 34_ConvTranspose3d_LayerNorm_GELU_Scaling

## Classification
- **Level**: 1 (Fusion)
- **Operation Count**: 4 operations
- **Operations**: ConvTranspose3d, LayerNorm, GELU, Scaling

## Original Model
```python
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, eps=1e-5, scaling_factor=1.0):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.layer_norm(x)
        x = torch.nn.functional.gelu(x)
        x = x * self.scaling_factor
        return x
```

## Data Flow
```
Input: [2, 2, 2, 4, 4]  (B, C_in, D, H, W)
    |
    v
ConvTranspose3d(2 -> 4, kernel=1, stride=1, pad=0) --> [2, 4, 2, 4, 4]
    |
    v
LayerNorm(4, eps=1e-5)  (normalizes last dim W=4) --> [2, 4, 2, 4, 4]
    |
    v
GELU activation --> [2, 4, 2, 4, 4]
    |
    v
Scale by 2.0 --> [2, 4, 2, 4, 4]
    |
    v
Output: [2, 4, 2, 4, 4]
```

## Decomposition Structure

### Level 0 - Kernels
1. **conv_transpose3d** (`level_0_kernel/conv_transpose3d.py`): `nn.ConvTranspose3d(2, 4, 1, stride=1, padding=0)` - 3D transposed convolution
2. **layer_norm** (`level_0_kernel/layer_norm.py`): `nn.LayerNorm(4, eps=1e-5)` - Layer normalization over last dimension
3. **gelu** (`level_0_kernel/gelu.py`): `F.gelu(x)` - GELU activation function (no learnable parameters)
4. **scaling** (`level_0_kernel/scaling.py`): `x * scaling_factor` - Constant scaling (no learnable parameters)

### Level 1 - Fusion
- **conv_norm_gelu_scale** (`level_1_fusion/conv_norm_gelu_scale.py`): Complete fused operation

## LayerNorm Dimension Analysis
This model has a subtle dimensional constraint. `nn.LayerNorm(out_channels)` normalizes over the last dimension of the tensor. After `ConvTranspose3d`, the output shape is `(B, C, D, H, W)`, so LayerNorm normalizes over `W`, not `C`.

For the model to work correctly, the last spatial dimension `W'` must equal `out_channels`. In the original model: `W' = (W-1)*stride - 2*padding + kernel_size = (32-1)*2 - 2*1 + 4 = 64 = out_channels`. For test sizes, we use `kernel_size=1, stride=1, padding=0` so `W' = W = 4 = out_channels`.

## Test Dimensions
| Parameter | Original | Test |
|-----------|----------|------|
| batch_size | 32 | 2 |
| in_channels | 32 | 2 |
| out_channels | 64 | 4 |
| D, H, W | 16, 32, 32 | 2, 4, 4 |
| kernel_size | 4 | 1 |
| stride | 2 | 1 |
| padding | 1 | 0 |
| bias | True | True |
| eps | 1e-5 | 1e-5 |
| scaling_factor | 1.0 | 2.0 |

## Notes
- The GELU and scaling kernels have no learnable parameters, making weight transfer straightforward.
- The dimensional constraint between LayerNorm and ConvTranspose3d spatial output is critical for correctness.
- The test uses `scaling_factor=2.0` (instead of the original 1.0) to ensure the scaling operation has a visible effect in tests.

## Verification
- Each kernel has independent shape and value tests via `run_tests()`.
- `verification/composition_test.py` confirms that chaining the four kernels produces identical output to the fused model with shared weights.
