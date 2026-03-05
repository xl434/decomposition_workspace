# Decomposition Analysis: 29_Matmul_Mish_Mish

## Classification
- **Level**: 1 (Fusion)
- **Operation count**: 3
- **Operations**: Linear, Mish, Mish

## Original Model
A linear layer followed by two consecutive Mish activations.
- Original sizes: batch=1024, in_features=8192, out_features=8192
- Test sizes: batch=2, in_features=16, out_features=32

## Decomposition Strategy
The model is a simple sequential pipeline of three operations, each
decomposed into its own Level-0 kernel:

1. **linear** - `nn.Linear(in_features, out_features)`: matrix multiply + bias
2. **mish_1** - First `F.mish`: element-wise Mish activation
3. **mish_2** - Second `F.mish`: element-wise Mish activation

The Level-1 fusion (`linear_mish_mish`) combines all three into a single
forward pass, matching the original model exactly.

## Data Flow
```
Input [2, 16]
  -> Linear -> [2, 32]
  -> Mish   -> [2, 32]
  -> Mish   -> [2, 32]
Output [2, 32]
```

## Verification
The composition test confirms that chaining the three Level-0 kernels
produces numerically identical output to the fused Level-1 model
(tolerance: rtol=1e-4, atol=1e-5).
