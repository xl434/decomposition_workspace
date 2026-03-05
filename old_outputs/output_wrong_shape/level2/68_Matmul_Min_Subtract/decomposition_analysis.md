# Decomposition Analysis: 68_Matmul_Min_Subtract

## Classification
- **Level**: 1 (Fusion)
- **Operation Count**: 3 operations
- **Operations**: Linear, Min, Subtract

## Original Model
```python
class Model(nn.Module):
    def __init__(self, in_features, out_features, constant):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))

    def forward(self, x):
        x = self.linear(x)
        x = torch.min(x, self.constant)
        x = x - self.constant
        return x
```

## Data Flow
```
Input: [2, 16]
    |
    v
Linear(16 -> 32)  --> [2, 32]
    |
    v
Min(x, constant=2.0) --> [2, 32]
    |
    v
Subtract(constant=2.0) --> [2, 32]
    |
    v
Output: [2, 32]
```

## Decomposition Structure

### Level 0 - Kernels
1. **linear** (`level_0_kernel/linear.py`): `nn.Linear(16, 32)` - Matrix multiply + bias
2. **min_clamp** (`level_0_kernel/min_clamp.py`): `torch.min(x, constant)` - Element-wise min clamping
3. **subtract** (`level_0_kernel/subtract.py`): `x - constant` - Element-wise subtraction

### Level 1 - Fusion
- **linear_min_subtract** (`level_1_fusion/linear_min_subtract.py`): Complete fused operation

## Shared Parameters
- The `constant` parameter (scalar, value=2.0) is shared between the `min_clamp` and `subtract` kernels.
- In the original model, this is a single `nn.Parameter`. When decomposed, each kernel holds its own copy, and the composition test ensures they stay synchronized.

## Test Dimensions
| Parameter | Original | Test |
|-----------|----------|------|
| batch_size | 128 | 2 |
| in_features | 16384 | 16 |
| out_features | 16384 | 32 |
| constant | 2.0 | 2.0 |

## Semantic Notes
- After `min(x, constant) - constant`, all output values are guaranteed to be <= 0.
- The min operation acts as an upper clamp, and the subsequent subtraction shifts the clamped range downward.
- This pattern is common in activation functions and value clipping scenarios.

## Verification
- Each kernel has independent shape and value tests via `run_tests()`.
- `verification/composition_test.py` confirms that chaining the three kernels produces identical output to the fused model with shared weights.
