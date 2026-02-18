# Hierarchical Model Decomposition Prompt

## Task Overview

You are tasked with hierarchically decomposing a PyTorch model into its constituent components, from high-level architectural modules down to individual kernel-level operations. This decomposition creates a tree of standalone, executable PyTorch test files that can be independently verified and optimized.

**Critical Requirement**: The decomposition must be VERIFIABLE - composing all leaf-level components must reproduce the exact output of the original model.

---

## Abstraction Levels

Decompose the model into these four abstraction levels:

| Level | Name | Description | Examples |
|-------|------|-------------|----------|
| 3 | Model | Complete models with multiple layers/stages | Transformer, VisionEncoder, LLM, VLM |
| 2 | Layer | Higher-level building blocks, typically repeated | TransformerBlock, AttentionLayer, MLPBlock |
| 1 | Fusion | Small groups (2-5) of operations commonly fused | Conv+BN+ReLU, QKV Projection, SwiGLU |
| 0 | Kernel | Single atomic operations | Conv2d, Linear, ReLU, LayerNorm, MatMul |

### Decomposition Flow
```
MODEL ──decompose──> LAYERS ──decompose──> FUSIONS ──decompose──> KERNELS
```

**Key Principle**: Each level decomposes ONLY to the next level down, never skipping levels.

---

## Phase 1: Analysis

### Step 1.1: Understand the Model Architecture

Read the model carefully and create an architecture map:

```markdown
## Architecture Analysis

### Module Hierarchy
- Model
  - self.embedding: nn.Embedding(vocab_size, hidden_dim)
  - self.layers: nn.ModuleList of N TransformerBlocks
    - self.attention: MultiHeadAttention
      - self.qkv_proj: nn.Linear(hidden_dim, 3*hidden_dim)
      - self.out_proj: nn.Linear(hidden_dim, hidden_dim)
    - self.mlp: MLP
      - self.up_proj: nn.Linear(hidden_dim, 4*hidden_dim)
      - self.down_proj: nn.Linear(4*hidden_dim, hidden_dim)
    - self.norm1: nn.LayerNorm(hidden_dim)
    - self.norm2: nn.LayerNorm(hidden_dim)
  - self.output_head: nn.Linear(hidden_dim, vocab_size)

### Data Flow (with shapes)
**NOTE**: Use the EXACT dimensions from the original model's declarations and get_inputs().
Do NOT reduce batch size, hidden dims, sequence length, or any other dimension.

Input: [batch, seq_len] (int64 - token indices)  ← use original get_inputs() values
  → embedding: [batch, seq_len] → [batch, seq_len, hidden_dim]
  → layer_0.norm1: [batch, seq_len, hidden_dim] → [batch, seq_len, hidden_dim]
  → layer_0.attention: [batch, seq_len, hidden_dim] → [batch, seq_len, hidden_dim]
  → layer_0.residual_add: + → [batch, seq_len, hidden_dim]
  → ... (repeat for each layer)
  → output_head: [batch, seq_len, hidden_dim] → [batch, seq_len, vocab_size]
Output: [batch, seq_len, vocab_size] (float32 - logits)
```

### Step 1.2: Classify Current Abstraction Level

Count operations in forward():
- **1 operation** → Level 0 (Kernel) - STOP, no decomposition needed
- **2-5 operations** → Level 1 (Fusion)
- **6-15 operations** with clear sub-modules → Level 2 (Layer)
- **15+ operations** or loops over layers → Level 3 (Model)

### Step 1.3: Identify Decomposition Targets

For the CURRENT level, identify what to extract at the NEXT level down:

| Current Level | Extract As | What to Look For |
|--------------|------------|------------------|
| Model (L3) | Layers (L2) | Repeated blocks, major sub-modules (encoder, decoder) |
| Layer (L2) | Fusions (L1) | Attention block, MLP block, normalization groups |
| Fusion (L1) | Kernels (L0) | Individual operations (linear, activation, norm) |

---

## Phase 2: Decomposition

### Step 2.1: Extract Components with Shape Tracking

For each component, track:
1. **Input shapes** - exact tensor dimensions entering this component
2. **Output shapes** - exact tensor dimensions leaving this component
3. **Intermediate shapes** - shapes at each step within the component
4. **Weight shapes** - dimensions of any learnable parameters

Create a **Component Specification**:

```python
# Component: attention_block
# Parent: transformer_layer_0
# Level: fusion (L1)
#
# Inputs:
#   x: [batch=2, seq=32, hidden=768] dtype=float32
#
# Weights:
#   qkv_proj.weight: [2304, 768]  # 3*768=2304
#   qkv_proj.bias: [2304]
#   out_proj.weight: [768, 768]
#   out_proj.bias: [768]
#
# Internal Flow:
#   x [2, 32, 768]
#   → qkv_proj [2, 32, 2304]
#   → reshape [2, 32, 3, 12, 64]  # 3 for q,k,v; 12 heads; 64 head_dim
#   → permute [3, 2, 12, 32, 64]  # separate q,k,v
#   → q,k,v each [2, 12, 32, 64]
#   → attention scores [2, 12, 32, 32]
#   → softmax [2, 12, 32, 32]
#   → weighted sum [2, 12, 32, 64]
#   → reshape [2, 32, 768]
#   → out_proj [2, 32, 768]
#
# Output:
#   output: [2, 32, 768] dtype=float32
```

### Step 2.2: Generate Component Files

Each file MUST follow this exact format:

```python
"""
Component: {descriptive_name}
Abstraction Level: {kernel|fusion|layer|model}
Parent: {parent_component_name or "root"}
Children: {list of child components if any}

Operations: {list of operations performed}

Input Shapes:
  - x: {shape} dtype={dtype}
  - (additional inputs if any)

Output Shapes:
  - output: {shape} dtype={dtype}

Weight Shapes:
  - {param_name}: {shape}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    {Brief description}

    Extracted from: {parent component}
    """
    def __init__(self):
        super().__init__()
        # Initialize with CORRECT shapes for weights
        # Weights must match input dtype when forward() is called

    def forward(self, x):
        # Ensure weight dtype matches input
        if hasattr(self, 'linear') and x.dtype != self.linear.weight.dtype:
            self.linear = self.linear.to(x.dtype)

        # Forward pass - MUST match exact computation of original
        return output

def get_inputs():
    """Generate test inputs with EXACT shapes from the ORIGINAL model's get_inputs().
    DO NOT reduce or modify dimensions — use the original model's exact values."""
    return [
        torch.randn(2, 32, 768, dtype=torch.float32),  # Must match original model
    ]

def get_init_inputs():
    """Return initialization parameters."""
    return []

# =============================================================================
# VERIFICATION SECTION
# =============================================================================

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    return [(2, 32, 768)]  # List of expected shapes

def run_tests():
    """Verify this component executes correctly."""
    try:
        model = Model(*get_init_inputs())
        model.eval()

        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)

            # 1. Basic validation
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            # 2. Shape validation
            expected_shapes = get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]

            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Output {i} shape mismatch: got {actual}, expected {expected}"

            # 3. Dtype validation
            expected_dtype = inputs[0].dtype
            if isinstance(output, tuple):
                for o in output:
                    assert o.dtype == expected_dtype, f"Dtype mismatch: {o.dtype} vs {expected_dtype}"
            else:
                assert output.dtype == expected_dtype, f"Dtype mismatch: {output.dtype} vs {expected_dtype}"

            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
            print("PASS")
            return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
```

---

## Phase 3: Verification Protocol

### Step 3.1: Individual Component Verification

For EACH generated component, verify:

1. **Execution Test**: `python {component}.py` runs without error
2. **Shape Test**: Output shapes match specification
3. **Dtype Test**: Output dtype matches input dtype
4. **NaN/Inf Test**: No numerical issues

### Step 3.2: Composition Verification (CRITICAL)

This is the key verification step - prove that the decomposition is correct by recomposing.

Create a **verification/composition_test.py** file that:
1. Imports the original model
2. Imports all decomposed components
3. Composes the components in the same order as the original
4. Compares outputs: `torch.allclose(original, composed, rtol=1e-4, atol=1e-5)`

### Step 3.3: Shape Flow Verification

Create a **verification/shape_flow_test.py** that traces tensor shapes through the decomposition.

### Step 3.4: Operation Coverage Verification

Ensure every operation from the original appears in exactly one kernel-level component.

---

## Phase 4: Output Structure

```
{output_dir}/
├── level_3_model/
│   └── {model_name}.py
├── level_2_layer/
│   ├── transformer_block_0.py
│   └── ...
├── level_1_fusion/
│   ├── attention_block.py
│   ├── mlp_block.py
│   └── ...
├── level_0_kernel/
│   ├── embedding.py
│   ├── linear_qkv.py
│   ├── layer_norm.py
│   └── ...
├── verification/
│   ├── composition_test.py
│   ├── shape_flow_test.py
│   └── operation_coverage.py
├── decomposition_tree.json
└── decomposition_analysis.md
```

---

## Constraints

### You MUST NOT:
1. Copy the entire model as a single "kernel"
2. Skip abstraction levels (model → kernel without layer/fusion)
3. Combine unrelated operations into fake "fusions"
4. Invent operations not in the original
5. Generate files that don't execute
6. **Modify, reduce, or simplify the dimensions declared in the original model** — Do NOT shrink batch size, channel counts, hidden dimensions, sequence lengths, or any other parameter for "faster testing". Use the EXACT values from the original model's declarations and `get_inputs()`/`get_init_inputs()` functions

### You MUST:
1. Trace complete data flow with shapes at each step
2. Preserve all dtypes (float32, bfloat16, etc.)
3. **Preserve the EXACT dimensions from the original model** — Use the same batch size, channel counts, hidden dimensions, kernel sizes, sequence lengths, and all other parameters as declared in the original model's `__init__()`, `get_inputs()`, and `get_init_inputs()`. Never reduce or simplify dimensions
4. Create composition_test.py that PASSES
5. Verify every component executes with correct shapes
6. Document reasoning for each decomposition decision

---

## Execution Checklist

Before submitting, verify:

- [ ] All component files execute without error
- [ ] `composition_test.py` PASSES (composed output matches original)
- [ ] `decomposition_tree.json` is complete
- [ ] Every leaf node is at kernel level (L0)
- [ ] Abstraction level hierarchy is respected (L3 → L2 → L1 → L0)

---

## Your Task

Given the model file below, perform hierarchical decomposition following all guidelines above.

**Model to Decompose:**
```python
{INSERT_MODEL_CODE_HERE}
```

**Output Directory:** `{output_directory}`

Please provide:
1. **Architecture Analysis** - Module hierarchy and data flow with shapes
2. **Decomposition Tree** (JSON format)
3. **All Component Files** - With complete, executable code
4. **Verification Files** - composition_test.py
5. **Verification Results** - Confirmation that all tests pass

Begin your decomposition:
