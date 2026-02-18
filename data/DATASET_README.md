# Dataset: KernelBench

This directory contains the KernelBench dataset for hierarchical decomposition testing.

## Structure

```
data/
├── kernelbench/
│   ├── level1/     # 100 kernel-level operations (already atomic)
│   ├── level2/     # 100 fusion-level operations (2-5 ops each)
│   └── level3/     # 51 model/layer-level (full decomposition needed)
├── manifest.txt    # List of all files
└── DATASET_README.md
```

## Statistics

| Level | Count | Description | Expected Decomposition |
|-------|-------|-------------|----------------------|
| Level 1 | 100 | Single kernel ops | None (already atomic) |
| Level 2 | 100 | Fused operations | Kernel extraction |
| Level 3 | 51 | Full models/layers | Full hierarchical |
| **Total** | **251** | | |

## Level 1 - Kernel Operations

These are already atomic operations. Decomposition should:
- Classify as `kernel` level
- Generate single test file
- No further decomposition

Examples:
- `1_Square_matrix_multiplication_.py` - MatMul
- `19_ReLU.py` - ReLU activation
- `23_Softmax.py` - Softmax
- `37_FrobeniusNorm.py` - Reduction operation

## Level 2 - Fusion Operations

These contain 2-5 fused operations. Decomposition should:
- Classify as `fusion` level
- Extract individual kernels
- Generate 2-5 kernel-level tests

Examples:
- Conv + BN + ReLU fusions
- Linear + Activation fusions
- Attention score computation

## Level 3 - Model/Layer Operations

These are complete models or layers. Full decomposition:
- Classify as `model` or `layer` level
- Extract layers, fusions, then kernels
- May generate 10-50+ kernel-level tests

Examples:
- `gpt_oss.py` - Full GPT model with MoE
- Transformer blocks
- Vision encoders

## Usage

### Single File Decomposition
```
# Give agent:
# 1. The prompt: prompts/MAIN_PROMPT.md
# 2. The model: data/kernelbench/level3/gpt_oss.py
# 3. Output dir: output/level3/gpt_oss/
```

### Batch Processing
```
# Use scripts/batch_decompose.py to process multiple files
python scripts/batch_decompose.py --level 3 --start 0 --count 10
```

## File Format

All KernelBench files follow this format:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ...

    def forward(self, x):
        # ...
        return output

def get_inputs():
    """Returns test inputs."""
    return [torch.randn(...)]

def get_init_inputs():
    """Returns __init__ arguments."""
    return [...]
```

## Notes

1. **External Dependencies**: Some Level 3 models have external imports.
   The agent should extract the core computation and make it self-contained.

2. **Shape Variations**: Some models use symbolic dimensions.
   Use concrete test values (batch=2, seq=32, etc.)

3. **Dtype Handling**: Many models use bfloat16.
   Ensure kernel tests handle dtype correctly.
