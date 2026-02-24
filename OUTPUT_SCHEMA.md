# Output Organization Schema

This document defines the standard organization for decomposition outputs.
All agents MUST follow this structure for consistency.

## Directory Structure

```
output/
├── {source_level}/                    # e.g., level1, level2, level3
│   └── {model_name}/                  # e.g., 1_conv2d, 50_matmul, gpt_oss
│       ├── decomposition_tree.json    # Hierarchy metadata
│       ├── decomposition_analysis.md  # Agent's analysis
│       ├── level_3_model/             # Model-level components (if applicable)
│       │   └── {model_name}.py
│       ├── level_2_layer/             # Layer-level components
│       │   ├── {layer_name}_0.py
│       │   └── {layer_name}_1.py
│       ├── level_1_fusion/            # Fusion-level components
│       │   ├── {fusion_name}_0.py
│       │   └── {fusion_name}_1.py
│       ├── level_0_kernel/            # Kernel-level components (REQUIRED)
│       │   ├── {kernel_name}_0.py
│       │   ├── {kernel_name}_1.py
│       │   └── ...
│       ├── steps/                     # Step-by-step intermediate results
│       │   ├── step_1_model_to_layers/
│       │   │   ├── original.py       # Copy of parent component
│       │   │   ├── refactored.py     # Parent rewritten with child calls ONLY
│       │   │   ├── children/         # Child component files
│       │   │   ├── weight_map.json   # Optional explicit param mapping
│       │   │   ├── verification_result.json  # verify_step.py output
│       │   │   └── coverage_report.json      # extract_ops.py output
│       │   ├── step_2_{layer}_to_fusions/
│       │   │   └── (same structure)
│       │   └── step_3_{fusion}_to_kernels/
│       │       └── (same structure)
│       └── verification/              # Verification tests
│           ├── composition_test.py
│           ├── test_results.json
│           ├── step_verification_summary.json  # Aggregated step results
│           └── coverage_summary.json           # Full coverage report
```

## Naming Conventions

### Directory Names
- Use lowercase with underscores
- Match the input file name (without .py extension)
- Examples: `gpt_oss`, `1_conv2d`, `50_matmul`

### Component File Names
Format: `{operation}_{shape_signature}_{dtype}.py`

Examples:
```
linear_768x3072_fp32.py
conv2d_3x64x224x224_fp32.py
layer_norm_768_fp32.py
softmax_2x12x32x32_fp32.py
matmul_2x12x32x64_2x12x64x32_fp32.py
```

For complex names, use descriptive prefixes:
```
attention_qkv_proj_768x2304_fp32.py
mlp_up_proj_768x3072_fp32.py
transformer_block_0_fp32.py
```

### Shape Signature Format
- Single tensor: `{dim1}x{dim2}x...`
- Multiple tensors: `{shape1}_{shape2}`
- Batch dimension: Include if fixed, omit if variable

### Data Type Codes
- `fp32` = float32
- `fp16` = float16
- `bf16` = bfloat16
- `i64` = int64
- `i32` = int32

## decomposition_tree.json Schema

```json
{
  "metadata": {
    "source_file": "data/kernelbench/level3/gpt_oss.py",
    "source_level": "level3",
    "model_name": "gpt_oss",
    "timestamp": "2024-01-15T10:30:00Z",
    "total_components": 42,
    "by_level": {
      "model": 1,
      "layer": 4,
      "fusion": 12,
      "kernel": 25
    },
    "verification_status": "PASSED"
  },
  "nodes": {
    "root": {
      "id": "root",
      "name": "GPT_OSS",
      "level": "model",
      "path": "level_3_model/gpt_oss.py",
      "children": ["layer_0", "layer_1", "..."],
      "input_shapes": {"x": [2, 32]},
      "output_shapes": {"logits": [2, 32, 50000]},
      "input_dtypes": {"x": "int64"},
      "output_dtypes": {"logits": "float32"}
    }
  },
  "data_flow": [
    {
      "from": "root",
      "to": "embedding",
      "tensor_name": "x",
      "shape": [2, 32],
      "dtype": "int64"
    }
  ]
}
```

## Dimension Preservation Rule

**CRITICAL**: All component files MUST use the EXACT dimensions from the original model.

- `get_inputs()` must return tensors with the same shapes as the original model's `get_inputs()`
- `get_init_inputs()` must pass the same initialization parameters as the original model
- Do NOT reduce batch size, channel counts, hidden dimensions, sequence lengths, or any other parameter for "faster testing" or "smaller examples"
- The shape signatures in file names must reflect the original model's actual dimensions

## Component File Template

Every component file MUST include:

```python
"""
Component: {name}
Source: {path to original model}
Abstraction Level: {kernel|fusion|layer|model}
Parent: {parent component name}

Operations: [{list of operations}]

Input Shapes:
  - {input_name}: {shape} dtype={dtype}

Output Shapes:
  - {output_name}: {shape} dtype={dtype}

Weight Shapes:
  - {weight_name}: {shape}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # ...

    def forward(self, x):
        # ...
        return output

def get_inputs():
    return [torch.randn(...)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(shape_tuple)]

def run_tests():
    # Standard test function
    pass

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
```

## Verification Requirements

### composition_test.py MUST:
1. Import the original model
2. Import all kernel components
3. Compose kernels to recreate original computation
4. Compare outputs with tolerance (rtol=1e-4, atol=1e-5)
5. Print "PASS" or "FAIL" with details

### test_results.json Format:
```json
{
  "timestamp": "2024-01-15T10:35:00Z",
  "total_components": 25,
  "passed": 25,
  "failed": 0,
  "composition_test": "PASSED",
  "max_difference": 1.2e-6,
  "component_results": [
    {"file": "level_0_kernel/linear_768x3072_fp32.py", "status": "PASS"},
    {"file": "level_0_kernel/gelu_fp32.py", "status": "PASS"}
  ]
}
```

## Example Output

For `data/kernelbench/level3/gpt_oss.py`:

```
output/level3/gpt_oss/
├── decomposition_tree.json
├── decomposition_analysis.md
├── level_3_model/
│   └── gpt_oss.py
├── level_2_layer/
│   ├── transformer_block_0.py
│   └── transformer_block_1.py
├── level_1_fusion/
│   ├── attention_block_fp32.py
│   ├── mlp_block_fp32.py
│   └── moe_block_fp32.py
├── level_0_kernel/
│   ├── embedding_201088x2880_bf16.py
│   ├── rms_norm_2880_fp32.py
│   ├── linear_qkv_2880x5760_bf16.py
│   ├── rotary_embedding_64_bf16.py
│   ├── sdpa_2x64x32x64_bf16.py
│   ├── linear_proj_4096x2880_bf16.py
│   ├── topk_k4_bf16.py
│   ├── softmax_dim1_bf16.py
│   ├── swiglu_2880_bf16.py
│   ├── add_residual_2x32x2880_bf16.py
│   └── ...
└── verification/
    ├── composition_test.py
    ├── test_results.json
    ├── step_verification_summary.json
    └── coverage_summary.json
```

## Steps Directory Schema

Each decomposition step saves intermediate results for verification.

### Step Directory Structure
```
steps/step_{N}_{component_name}/
  original.py              # Copy of parent component being decomposed
  refactored.py            # Parent rewritten using ONLY child module calls
  children/                # Child component files for this step
    child_0.py
    child_1.py
  weight_map.json          # Optional: explicit parameter name mapping
  verification_result.json # Output of scripts/verify_step.py
  coverage_report.json     # Output of scripts/extract_ops.py
```

### Refactored Code Constraints

The `refactored.py` file MUST follow these anti-cheat rules:
1. All `nn.Parameter`s must belong to child submodules (no standalone weights)
2. `forward()` may ONLY contain:
   - Child module calls: `self.child_a(x)`
   - Data plumbing: `x + residual`, `torch.cat(...)`, `torch.split(...)`
   - Shape ops: `x.reshape(...)`, `x.permute(...)`, `x.view(...)`
   - Indexing: `x[:, 0]`, `q, k, v = x.chunk(3)`
3. `forward()` must NOT contain: `F.relu`, `F.softmax`, `torch.matmul`, `nn.Linear`, or any other compute ops

### verification_result.json Schema
```json
{
  "timestamp": "2026-02-20T10:00:00",
  "original_file": "steps/step_1/original.py",
  "refactored_file": "steps/step_1/refactored.py",
  "status": "PASS",
  "anticheat": {
    "status": "PASS",
    "source_violations": [],
    "parameter_violations": []
  },
  "weight_transfer": {
    "status": "OK",
    "mapped_count": 10,
    "unmapped_original": [],
    "unmapped_refactored": []
  },
  "numerical_comparison": {
    "status": "PASS",
    "rtol": 1e-5,
    "atol": 1e-6,
    "input_dtype": "torch.float32",
    "num_trials": 3,
    "max_diff_across_trials": 2.384e-07,
    "trials": [
      {"trial": 0, "seed": 42, "max_diff": 2.384e-07, "pass": true}
    ]
  }
}
```

### coverage_report.json Schema
```json
{
  "timestamp": "2026-02-20T10:00:00",
  "model_file": "steps/step_1/original.py",
  "status": "FULL_COVERAGE",
  "extraction_method": "torch.fx.symbolic_trace",
  "coverage": {
    "total_original_ops": 5,
    "total_decomposed_ops": 5,
    "covered_op_count": 5,
    "coverage_pct": 100.0,
    "missing_op_types": [],
    "extra_op_types": [],
    "op_details": [
      {"op_type": "Conv2d", "original_count": 2, "decomposed_count": 2, "status": "covered"}
    ]
  }
}
```

### step_verification_summary.json Schema
```json
{
  "timestamp": "2026-02-20T10:05:00",
  "decomp_dir": "output/level3/11_VGG16/",
  "status": "PASS",
  "total_steps": 15,
  "passed": 15,
  "failed": 0,
  "skipped": 0,
  "steps": [
    {
      "step_name": "step_1_model_to_layers",
      "verification_status": "PASS",
      "coverage_pct": 100.0,
      "max_diff": 2.384e-07,
      "anticheat_status": "PASS"
    }
  ]
}
```
