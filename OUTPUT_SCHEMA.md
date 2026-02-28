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

**Each file represents a unique workload**: a specific operation with specific input/output dimensions and dtype. The **file name** encodes the workload identity (operation + shape + dtype), while the **module itself** uses `get_init_inputs()` to pass dimensions — so it's testable with standard `Model(*get_init_inputs())`. The `get_inputs()` function returns tensors with the correct shapes for that specific workload.

Examples:
```
linear_768x3072_fp32.py       # Linear: in=768, out=3072, float32
linear_3072x768_fp32.py       # Linear: in=3072, out=768 (DIFFERENT workload)
conv2d_3x768_16x16_fp32.py    # Conv2d: in_ch=3, out_ch=768, kernel=16x16
layer_norm_768_fp32.py         # LayerNorm: normalized_shape=768
rms_norm_960_fp32.py           # RMSNorm: hidden_size=960
softmax_1x12x1024x1024_fp32.py
matmul_1x12x1024x64_1x12x64x1024_fp32.py
embedding_49280x960_i64_fp32.py  # Embedding: vocab=49280, dim=960, input=int64, output=float32
silu_fp32.py                   # Elementwise ops without shape-dependent params
gelu_fp32.py                   # (shape-independent — one file covers all shapes)
```

**Shape signature rules:**
- Include dimensions that define the workload identity (weight shapes, not batch/sequence)
- For **parameterized ops** (Linear, Conv2d, Embedding): use parameter dimensions (e.g., `linear_768x3072` = weight shape)
- For **elementwise ops** (ReLU, SiLU, GELU): omit shape if the op is shape-independent. Use `silu_fp32.py` not `silu_1x50x720_fp32.py`
- For **reduction ops** with fixed dims (Softmax, LayerNorm): include the relevant dimension

### Kernel Identity & Deduplication

A **unique workload** is defined by: `(operation, parameter_shapes, input_dtype, output_dtype)`.

**Two kernels are the SAME workload** (share one file) if all of these match:
- Same operation type (e.g., both are Linear)
- Same parameter shapes (e.g., both have weight [768, 3072])
- Same input/output dtypes

**Two kernels are DIFFERENT workloads** (separate files) when ANY differs:
- Different parameter shapes: `linear_768x3072_fp32.py` vs `linear_3072x768_fp32.py`
- Different dtypes: `linear_768x3072_fp32.py` vs `linear_768x3072_bf16.py`
- Different operations: `relu_fp32.py` vs `gelu_fp32.py`

**Rules:**
- **One file per unique workload** — dimensions hardcoded, no shape parameters in `__init__`
- **No `_2`, `_3` suffixes** — do NOT create `relu_fp32.py` and `relu_fp32_2.py`
- **Multiple instances import from the same file** — in `refactored.py`, create separate instances with different names
- Each instance gets its own weights via the weight map, but shares the kernel definition

**Example — kernel file with workload-specific inputs:**
```python
# level_0_kernel/linear_768x3072_fp32.py
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

def get_inputs():
    return [torch.randn(1, 128, 768)]   # input with last dim matching in_features

def get_init_inputs():
    return [768, 3072]                   # in_features, out_features for this workload
```

### Shape Signature Format
- Parameter dimensions: `{dim1}x{dim2}x...` (e.g., `768x3072` for a Linear weight)
- Multiple parameter groups: separate with `_` (e.g., `matmul_1x12x1024x64_1x12x64x1024`)
- Batch/sequence dimensions: **omit** from the signature (they don't define the workload)
- Elementwise ops: omit shape entirely if shape-independent (e.g., `silu_fp32.py`)

### Data Type Codes
- `fp32` = float32
- `fp16` = float16
- `bf16` = bfloat16
- `i64` = int64
- `i32` = int32
- For ops with mixed input/output dtypes (e.g., Embedding: int64 in, float32 out), use `{input_dtype}_{output_dtype}` or the output dtype

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
    "unique_kernels": 10,
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
    },
    "linear_qkv_layer0": {
      "id": "linear_qkv_layer0",
      "name": "QKV Projection (layer 0)",
      "level": "kernel",
      "path": "level_0_kernel/linear_qkv_2880x5760_bf16.py",
      "instance_count": 24,
      "children": [],
      "input_shapes": {"x": [2, 32, 2880]},
      "output_shapes": {"out": [2, 32, 5760]},
      "input_dtypes": {"x": "bfloat16"},
      "output_dtypes": {"out": "bfloat16"}
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
├── decomposition_log.json                   # Decision/difficulty log
├── level_3_model/
│   └── gpt_oss.py
├── level_2_layer/
│   ├── transformer_block.py
│   └── output_head.py
├── level_1_fusion/
│   ├── attention_block.py
│   ├── mlp_block.py
│   └── moe_block.py
├── level_0_kernel/                          # Deduplicated: one file per unique WORKLOAD
│   ├── embedding_201088x2880_bf16.py       # 1 instance (vocab=201088, dim=2880)
│   ├── rms_norm_2880_fp32.py               # 24 instances (hidden=2880)
│   ├── linear_2880x5760_bf16.py            # 24 instances (QKV proj)
│   ├── linear_4096x2880_bf16.py            # 24 instances (down proj)
│   ├── linear_2880x4096_bf16.py            # 24 instances (up proj - DIFFERENT workload)
│   ├── rotary_embedding_64_bf16.py         # 24 instances (head_dim=64)
│   ├── softmax_dim1_bf16.py                # 24 instances
│   ├── matmul_2x32x64_2x64x32_bf16.py     # 24 instances (attn scores)
│   ├── silu_bf16.py                        # 24 instances (elementwise, shape-independent)
│   └── ...                                 # Total: ~10 unique files, ~250 instances
└── verification/
    ├── composition_test.py
    ├── test_results.json
    ├── step_verification_summary.json
    └── coverage_summary.json               # REQUIRED: extract_ops.py output
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
