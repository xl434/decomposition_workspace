# Decomposition Analysis: 2_ShallowWideMLP

## Overview
- **Model**: 2_ShallowWideMLP from KernelBench Level 3
- **Decomposition Level**: Level 1 (Fusion of 5 Level 0 kernels)
- **Number of Operations**: 5

## Architecture
A shallow but wide multi-layer perceptron with two hidden layers. The original KernelBench model uses very large dimensions (input=16384, hidden=32768, output=16384); this decomposition uses small test dimensions (input=16, hidden=32, output=8) for verification.

## Operation Details

| # | Operation | Type       | Input Shape | Output Shape |
|---|-----------|------------|-------------|--------------|
| 1 | Linear_0  | nn.Linear  | [2, 16]     | [2, 32]      |
| 2 | ReLU_0    | nn.ReLU    | [2, 32]     | [2, 32]      |
| 3 | Linear_1  | nn.Linear  | [2, 32]     | [2, 32]      |
| 4 | ReLU_1    | nn.ReLU    | [2, 32]     | [2, 32]      |
| 5 | Linear_2  | nn.Linear  | [2, 32]     | [2, 8]       |

### Linear Parameters
- Linear_0: in=16, out=32
- Linear_1: in=32, out=32
- Linear_2: in=32, out=8

### Activation
- ReLU (no learnable parameters)

## Data Flow
```
x=[2,16]
   |
[Linear(16,32)]  -- linear_0
   |
[2,32]
   |
[ReLU]           -- relu_0
   |
[2,32]
   |
[Linear(32,32)]  -- linear_1
   |
[2,32]
   |
[ReLU]           -- relu_1
   |
[2,32]
   |
[Linear(32,8)]   -- linear_2
   |
[2,8]
```

## Decomposition Hierarchy
```
Level 1: mlp_pipeline (fusion)
  |-- Level 0: linear_0.py    (nn.Linear, 16->32)
  |-- Level 0: relu_0.py      (nn.ReLU)
  |-- Level 0: linear_1.py    (nn.Linear, 32->32)
  |-- Level 0: relu_1.py      (nn.ReLU)
  |-- Level 0: linear_2.py    (nn.Linear, 32->8)
```

## Weight Mapping
The original model uses `nn.Sequential` with numeric indices:
- `network.0` -> linear_0.fc
- `network.1` -> relu_0 (no weights)
- `network.2` -> linear_1.fc
- `network.3` -> relu_1 (no weights)
- `network.4` -> linear_2.fc

## File Structure
```
2_ShallowWideMLP/
  level_0_kernel/
    linear_0.py            -- First linear layer (16 -> 32)
    relu_0.py              -- First ReLU activation
    linear_1.py            -- Second linear layer (32 -> 32)
    relu_1.py              -- Second ReLU activation
    linear_2.py            -- Output linear layer (32 -> 8)
  level_1_fusion/
    mlp_pipeline.py        -- Fused model (all 5 operations)
  verification/
    composition_test.py    -- Verifies composed kernels match original
  decomposition_tree.json
  decomposition_analysis.md
```

## Notes
- The ReLU kernels are stateless (no learnable parameters), so no weight sharing is needed for them in composition tests.
- The original model wraps everything in `nn.Sequential`, making weight extraction straightforward via numeric indices.
- This is a purely sequential pipeline with no branching or skip connections.
