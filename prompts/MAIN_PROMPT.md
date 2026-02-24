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

## Phase 2: Step-by-Step Decomposition with Verification Gates

**CRITICAL: Decompose ONE level at a time. Verify EACH step before proceeding.**

Do NOT decompose all levels at once. Follow this iterative loop for each level transition.

### Step 2.1: Decompose to Next Level

For the CURRENT component, identify children at the next level down and create a file for each child component following the standard template (see Component File Format below).

For each component, track:
1. **Input shapes** - exact tensor dimensions entering this component
2. **Output shapes** - exact tensor dimensions leaving this component
3. **Intermediate shapes** - shapes at each step within the component
4. **Weight shapes** - dimensions of any learnable parameters

### Step 2.2: Create Refactored Code

For each component you decompose, create a **refactored version** that replaces inline computation with calls to the child modules you extracted. Save this as `steps/step_N_name/refactored.py`.

**Anti-Cheat Rules for Refactored Code:**

The refactored `forward()` may ONLY contain:

| Allowed (data plumbing) | Disallowed (must be in child module) |
|---|---|
| `self.child_module(x)` — child calls | `nn.Linear(...)`, `nn.Conv2d(...)` — module construction |
| `x + residual`, `x * scale` — arithmetic | `F.linear(...)`, `F.softmax(...)` — functional compute |
| `torch.cat(...)`, `torch.split(...)` — assembly | `torch.matmul(...)`, `torch.bmm(...)` — compute ops |
| `x.reshape(...)`, `x.permute(...)` — shape ops | `F.gelu(...)`, `F.relu(...)` — activations |
| `x.transpose(...)`, `x.view(...)` — shape ops | `F.layer_norm(...)`, `F.batch_norm(...)` — norms |
| `x[:, :, 0]` — indexing/slicing | Any `nn.Module` not imported from children |

**All parameters must belong to child submodules.** No standalone `nn.Linear`, `nn.Conv2d`, etc. in the refactored model.

**Example — VGG Block decomposed to fusions:**

Original `features_block_1.py`:
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, x):
        return self.block(x)
```

Refactored `steps/step_2_block1_to_fusions/refactored.py`:
```python
sys.path.insert(0, str(Path(__file__).parent / "children"))
from conv_relu_3x64 import Model as ConvRelu1
from conv_relu_64x64 import Model as ConvRelu2
from maxpool2d import Model as MaxPool

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_1 = ConvRelu1()   # child only
        self.conv_relu_2 = ConvRelu2()   # child only
        self.maxpool = MaxPool()          # child only

    def forward(self, x):
        x = self.conv_relu_1(x)          # child call
        x = self.conv_relu_2(x)          # child call
        x = self.maxpool(x)              # child call
        return x

def get_inputs():
    return [torch.randn(10, 3, 224, 224)]  # same as parent
```

**Example — Transformer block with residuals:**
```python
from children.norm_attention import Model as NormAttention
from children.norm_mlp import Model as NormMLP

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_attn = NormAttention()
        self.norm_mlp = NormMLP()

    def forward(self, x):
        residual = x                     # data plumbing: allowed
        x = self.norm_attn(x)            # child call
        x = x + residual                 # data plumbing: allowed
        residual = x                     # data plumbing: allowed
        x = self.norm_mlp(x)             # child call
        x = x + residual                 # data plumbing: allowed
        return x
```

### Step 2.3: Verify This Step (GATE)

Run the standard verification script:
```bash
python scripts/verify_step.py \
    --original path/to/parent.py \
    --refactored steps/step_N_name/refactored.py \
    --output steps/step_N_name/verification_result.json
```

This script automatically:
1. **Anti-cheat validation** — scans refactored code for disallowed ops and standalone parameters
2. **Weight transfer** — maps and copies weights from original to refactored model
3. **Numerical comparison** — runs both on same inputs (3 trials), compares outputs

**You MUST NOT proceed to the next level until this step PASSES.**

If verification fails:
- Check `verification_result.json` for `max_diff` and trial details
- Common issues: missing residual connection, wrong op order, missing weight in mapping, dtype mismatch
- Fix the refactored code or child components and re-run

If the standard script cannot handle your model (special components, unusual architecture), write a custom verification script that replicates the **exact same logic**: same tolerance rules, same weight transfer, same output JSON format, same anti-cheat checks.

### Step 2.4: Coverage Check (Optional but Recommended)

Run coverage analysis:
```bash
python scripts/extract_ops.py \
    --model path/to/parent.py \
    --children steps/step_N_name/children/*.py \
    --output steps/step_N_name/coverage_report.json
```

### Step 2.5: Repeat for Each Child

For each child that is NOT at kernel level (L0), repeat Steps 2.1-2.4.

**Decomposition order:**
1. Model (L3) → Layers (L2) — verify step
2. For each Layer: Layer (L2) → Fusions (L1) — verify each step
3. For each Fusion: Fusion (L1) → Kernels (L0) — verify each step

### Component File Format

Each component file MUST follow this format:

```python
"""
Component: {descriptive_name}
Abstraction Level: {kernel|fusion|layer|model}
Parent: {parent_component_name or "root"}
Children: {list of child components if any}

Operations: {list of operations performed}

Input Shapes:
  - x: {shape} dtype={dtype}

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

    def forward(self, x):
        # Forward pass - MUST match exact computation of original
        return output

def get_inputs():
    """Generate test inputs with EXACT shapes from the ORIGINAL model's get_inputs().
    DO NOT reduce or modify dimensions."""
    return [torch.randn(2, 32, 768, dtype=torch.float32)]

def get_init_inputs():
    """Return initialization parameters."""
    return []

def get_expected_output_shape():
    """Return expected output shape(s) for verification."""
    return [(2, 32, 768)]

def run_tests():
    """Verify this component executes correctly."""
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape] if isinstance(output, torch.Tensor) else [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Output {i} shape mismatch: got {actual}, expected {expected}"
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

## Phase 3: Final Verification (End-to-End Double-Check)

**Purpose**: The step-by-step verification in Phase 2 is the primary correctness guarantee — each level transition is proven equivalent with shared weights. Phase 3 provides an **independent end-to-end double-check**: build a single model from ONLY L0 kernel components and verify it matches the original. This serves as:

1. **Independent proof** — does not rely on step results; anyone can run it standalone
2. **Transitive chain validation** — catches any subtle issue that step-level verification might miss (e.g., weight mapping errors that happen to cancel out at one level but compound across levels)
3. **Standalone artifact** — a single file that proves the decomposition is correct, runnable without understanding the step-by-step flow

### Step 3.1: End-to-End Composition Test (REQUIRED)

Create **verification/composition_test.py** that:
1. Imports the **original model**
2. Imports **all kernel-level (L0) components** from `level_0_kernel/`
3. Builds a `ComposedModel` that chains all L0 kernels in the correct order to recreate the full computation (following the data flow from `decomposition_tree.json`)
4. Transfers weights from original → composed model
5. Compares outputs: `torch.allclose(original_output, composed_output, rtol=1e-4, atol=1e-5)`

The `ComposedModel.forward()` follows the same anti-cheat rules as refactored code — it should ONLY call kernel child modules plus data plumbing (residual adds, reshape, cat, etc.).

```bash
python {output_dir}/verification/composition_test.py
# Must print PASS
```

### Step 3.2: Coverage Summary

Run full coverage analysis comparing original model ops against **leaf-level (L0 kernel) components only**. Do NOT count ops from L1/L2 components — they contain the same ops nested inside them and would inflate the count.

```bash
python scripts/extract_ops.py \
    --model {original_model.py} \
    --decomp-dir {output_dir}/ \
    --output {output_dir}/verification/coverage_summary.json
```

> **Note**: `--decomp-dir` scans only `level_0_kernel/` to avoid double/triple-counting ops that appear at multiple hierarchy levels. If you need per-step coverage, use `--children` with explicit file paths instead.

### Step 3.3: Step Pipeline Summary (optional, for CI/human review)

Re-validate all steps and aggregate results:
```bash
python scripts/run_step_pipeline.py {output_dir}/
```
This is redundant if the agent already ran `verify_step.py` at each step in Phase 2. It exists for post-hoc validation by humans or CI systems.

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
├── steps/
│   ├── step_1_model_to_layers/
│   │   ├── original.py
│   │   ├── refactored.py
│   │   ├── children/
│   │   ├── verification_result.json
│   │   └── coverage_report.json
│   ├── step_2_{layer}_to_fusions/
│   │   └── ...
│   └── step_3_{fusion}_to_kernels/
│       └── ...
├── verification/
│   ├── composition_test.py
│   ├── step_verification_summary.json
│   └── coverage_summary.json
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
7. **Put compute ops in refactored code** — refactored forward() must ONLY call child modules + data plumbing. No `F.relu`, `torch.matmul`, `nn.Linear`, etc.
8. **Proceed to the next level before the current step's verify_step.py PASSES**

### You MUST:
1. Trace complete data flow with shapes at each step
2. Preserve all dtypes (float32, bfloat16, etc.)
3. **Preserve the EXACT dimensions from the original model** — Use the same batch size, channel counts, hidden dimensions, kernel sizes, sequence lengths, and all other parameters as declared in the original model's `__init__()`, `get_inputs()`, and `get_init_inputs()`. Never reduce or simplify dimensions
4. **Decompose one level at a time** with verification between each step
5. **Create refactored.py at each step** and run `verify_step.py` to confirm numerical equivalence
6. Create final composition_test.py that PASSES
7. Verify every component executes with correct shapes
8. Document reasoning for each decomposition decision

---

## Execution Checklist

Before submitting, verify:

- [ ] **Each decomposition step** has a refactored.py with ONLY child module calls
- [ ] **Each step's verify_step.py PASSES** (numerical equivalence confirmed)
- [ ] All component files execute without error (print "PASS")
- [ ] `composition_test.py` PASSES (final composed output matches original)
- [ ] `decomposition_tree.json` is complete
- [ ] Every leaf node is at kernel level (L0)
- [ ] Abstraction level hierarchy is respected (L3 → L2 → L1 → L0)
- [ ] Coverage analysis shows no missing operations
- [ ] ALL DIMENSIONS MATCH THE ORIGINAL MODEL EXACTLY

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
2. **Step-by-step decomposition** with refactored.py and verify_step.py results at each level
3. **Decomposition Tree** (JSON format)
4. **All Component Files** - With complete, executable code
5. **Verification Files** - step results + final composition_test.py
6. **Coverage Report** - extract_ops.py results

Begin your decomposition:
