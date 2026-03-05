# Decomposition Analysis: 45_Gemm_Sigmoid_LogSumExp

## Overview
- **Model**: 45_Gemm_Sigmoid_LogSumExp
- **Classification**: Level 1 (Fusion)
- **Total Operations**: 4
- **Decomposition Depth**: 1 level (Level 0 kernels composed into Level 1 fusion)

## Original Model
```python
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.logsumexp(x, dim=1)
        return x
```

## Production Dimensions
- batch_size=16384, input_size=2048, hidden_size=4096, output_size=1024

## Test Dimensions
- batch_size=2, input_size=16, hidden_size=32, output_size=8

## Data Flow (Test Dimensions)
```
Input: [2, 16]
  -> Linear1(16, 32)    -> [2, 32]
  -> Sigmoid             -> [2, 32]
  -> Linear2(32, 8)     -> [2, 8]
  -> LogSumExp(dim=1)   -> [8]
```

## Decomposition Tree

```
45_Gemm_Sigmoid_LogSumExp (Level 1 - Fusion)
├── linear_1 (Level 0 - Kernel) : nn.Linear(16, 32)
├── sigmoid  (Level 0 - Kernel) : torch.sigmoid
├── linear_2 (Level 0 - Kernel) : nn.Linear(32, 8)
└── logsumexp(Level 0 - Kernel) : torch.logsumexp(dim=1)
```

## Component Summary

| Component | Level | Type   | Input Shape | Output Shape | Parameters |
|-----------|-------|--------|-------------|-------------|------------|
| linear_1  | 0     | kernel | [2, 16]     | [2, 32]     | weight[32,16], bias[32] |
| sigmoid   | 0     | kernel | [2, 32]     | [2, 32]     | none |
| linear_2  | 0     | kernel | [2, 32]     | [2, 8]      | weight[8,32], bias[8] |
| logsumexp | 0     | kernel | [2, 8]      | [8]         | none |
| gemm_sigmoid_logsumexp | 1 | fusion | [2, 16] | [8] | all of above |

## Notes
- LogSumExp over dim=1 reduces across the batch dimension, collapsing [2, 8] to [8].
- The sigmoid activation is parameter-free and operates element-wise.
- Linear layers contain the only learnable parameters in this model.
- The sequential chain has no branching, making decomposition straightforward.
