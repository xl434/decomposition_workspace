# Decomposition Analysis: 38_LSTMBidirectional

## Overview
- **Model**: 38_LSTMBidirectional from KernelBench Level 3
- **Decomposition Level**: Level 1 (Fusion of 3 Level 0 kernels)
- **Number of Operations**: 3

## Architecture
The model performs bidirectional LSTM sequence processing, extracts the last timestep, and projects it through a fully connected layer for classification/regression.

## Operation Details

| # | Operation   | Type         | Input Shape(s)                            | Output Shape |
|---|-------------|--------------|-------------------------------------------|--------------|
| 1 | LSTM        | nn.LSTM      | x=[2,4,8], h0=[4,2,16], c0=[4,2,16]      | [2,4,32]     |
| 2 | Slice Last  | [:, -1, :]   | [2,4,32]                                  | [2,32]       |
| 3 | FC          | nn.Linear    | [2,32]                                    | [2,4]        |

### LSTM Parameters
- `input_size`: 8
- `hidden_size`: 16
- `num_layers`: 2
- `batch_first`: True
- `bidirectional`: True
- Output features: hidden_size * 2 = 32

### FC Parameters
- `in_features`: 32 (hidden_size * 2, due to bidirectional)
- `out_features`: 4

## Data Flow
```
x=[2,4,8], h0=[4,2,16], c0=[4,2,16]
                |
      [nn.LSTM bidirectional]
                |
          [2,4,32]
                |
        [slice [:,-1,:]]
                |
           [2,32]
                |
         [nn.Linear]
                |
            [2,4]
```

## Decomposition Hierarchy
```
Level 1: lstm_slice_fc (fusion)
  |-- Level 0: lstm.py          (nn.LSTM, bidirectional)
  |-- Level 0: slice_last.py    (tensor indexing, no weights)
  |-- Level 0: linear.py        (nn.Linear)
```

## File Structure
```
38_LSTMBidirectional/
  level_0_kernel/
    lstm.py                -- Bidirectional LSTM kernel
    slice_last.py          -- Last-timestep slicing kernel
    linear.py              -- Fully connected kernel
  level_1_fusion/
    lstm_slice_fc.py       -- Fused model (all 3 operations)
  verification/
    composition_test.py    -- Verifies composed kernels match original
  decomposition_tree.json
  decomposition_analysis.md
```

## Notes
- The slice operation `out[:, -1, :]` is a stateless kernel with no learnable parameters.
- `batch_first=True` means LSTM input/output shape is `[batch, seq, features]`.
- The bidirectional LSTM doubles both the h0/c0 first dimension and the output features dimension.
