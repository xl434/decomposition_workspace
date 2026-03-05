# Decomposition Analysis: 41_GRUBidirectional

## Overview
- **Model**: 41_GRUBidirectional from KernelBench Level 3
- **Decomposition Level**: Level 0 (Leaf Kernel)
- **Number of Operations**: 1

## Architecture
The model consists of a single `nn.GRU` layer with `bidirectional=True`. Since there is only one operation, this model IS a Level 0 kernel and requires no decomposition.

## Operation Details

| Operation | Type    | Input Shape(s)              | Output Shape  |
|-----------|---------|-----------------------------|---------------|
| GRU       | nn.GRU  | x=[4,2,8], h0=[4,2,16]     | [4,2,32]      |

### GRU Parameters
- `input_size`: 8
- `hidden_size`: 16
- `num_layers`: 2
- `bias`: True
- `batch_first`: False
- `bidirectional`: True

## Data Flow
```
x=[4,2,8], h0=[4,2,16]
        |
  [nn.GRU bidirectional]
        |
  output=[4,2,32]
```

## Key Differences from 39_GRU
- `bidirectional=True` doubles the number of directions (2 instead of 1).
- `h0` first dimension is `num_layers * 2 = 4` (instead of `num_layers = 2`).
- Output last dimension is `hidden_size * 2 = 32` (instead of `hidden_size = 16`).

## Notes on Original Code
The original KernelBench model includes `self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))` which references a global `batch_size` variable. This attribute is never used in `forward()` and is omitted from the kernel.

## File Structure
```
41_GRUBidirectional/
  level_0_kernel/
    gru_bidirectional.py   -- The bidirectional GRU kernel (the entire model)
  verification/
    composition_test.py    -- Verifies kernel matches original model
  decomposition_tree.json
  decomposition_analysis.md
```
