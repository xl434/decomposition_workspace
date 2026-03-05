# Decomposition Analysis: 56_Matmul_Sigmoid_Sum

## Classification
- **Level**: 1 (Fusion)
- **Operation count**: 3
- **Operations**: Linear, Sigmoid, Sum

## Original Model
A linear layer followed by sigmoid activation and a sum reduction over dim=1.
- Original sizes: batch=128, input_size=32768, hidden_size=32768
- Test sizes: batch=2, input_size=16, hidden_size=32

## Decomposition Strategy
The model is a sequential pipeline of three operations, each decomposed
into its own Level-0 kernel:

1. **linear** - `nn.Linear(input_size, hidden_size)`: matrix multiply + bias
2. **sigmoid** - `torch.sigmoid`: element-wise sigmoid activation
3. **sum** - `torch.sum(dim=1, keepdim=True)`: reduction over feature dim

The Level-1 fusion (`linear_sigmoid_sum`) combines all three into a single
forward pass, matching the original model exactly.

## Data Flow
```
Input [2, 16]
  -> Linear  -> [2, 32]
  -> Sigmoid -> [2, 32]
  -> Sum     -> [2, 1]
Output [2, 1]
```

## Notes
The sum reduction changes the output shape, collapsing the hidden dimension
to a single value per batch element. This makes the final output shape
[batch_size, 1] rather than [batch_size, hidden_size].

## Verification
The composition test confirms that chaining the three Level-0 kernels
produces numerically identical output to the fused Level-1 model
(tolerance: rtol=1e-4, atol=1e-5).
