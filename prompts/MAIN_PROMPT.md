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

**Key Principle**: Prefer decomposing to the **next level down** first. Extract the largest meaningful children as the primary decomposition targets. However, if a child component is semantically at a **lower level** (e.g., a single-op kernel sitting directly inside a layer), classify it at its natural level — do NOT wrap it in a fake intermediate level.

**Allowed parent → child level relationships:**
- Model (L3) → Layer (L2), Fusion (L1), Kernel (L0)
- Layer (L2) → Fusion (L1), Kernel (L0)
- Fusion (L1) → Kernel (L0)

**Priority**: Always look for next-level-down children first. Only classify a child at a lower level when it genuinely belongs there (e.g., a standalone `nn.Embedding` inside a model is a kernel, not a layer).

---

## Phase 0: Model Preparation (for multi-file / external-dependency models)

> **If your model is already a single self-contained `.py` file with `Model`, `get_inputs()`, and `get_init_inputs()`, skip to Phase 1.**

This phase converts a multi-file model, HuggingFace model, or repo with external dependencies into a single self-contained file that the rest of the pipeline can work with.

### Step 0.1: Explore and Map Dependencies

Starting from the entry file/class, systematically discover everything needed for the forward pass:

1. **Read the entry file** and identify the main model class and its `forward()` signature
2. **Trace all local imports recursively** — for each import in the entry file, read that file, find its imports, and continue until you reach only external packages (torch, transformers, etc.)
3. **Map the dependency graph**:
   - Which local files are needed? (e.g., `configuration_*.py`, helper modules)
   - Which external packages are needed? (e.g., `transformers`, `timm`, `einops`)
   - Which classes/functions from each file are actually used in the forward pass?
4. **Identify the forward() signature** — what are the input names, shapes, dtypes?
   - Read `get_inputs()`, test scripts, example usage, or docstrings to determine concrete input shapes
   - If shapes aren't documented, trace the code to infer them from layer dimensions
5. **Extract config values** — find the configuration class or defaults:
   - Read the config file to get all hyperparameters (hidden_size, num_layers, etc.)
   - Note which values are needed for model construction vs. training-only
6. **Classify what to keep vs. ignore**:
   - **Keep**: Everything in the forward computation graph (model layers, activations, norms, attention, projections)
   - **Ignore**: Tokenizers, processors, data loaders, training utilities, logging, config file I/O, checkpoint loading logic

Document your findings in `{output_dir}/decomposition_analysis.md` before proceeding.

### Step 0.2: Create a Self-Contained Wrapper

Create `{output_dir}/level_3_model/{model_name}.py` that:

1. **Inlines or imports all needed code** — Copy essential classes/functions from dependency files directly into one file (or a local package folder)

2. **Hardcodes the config** — Replace config-file loading with concrete values:
   ```python
   # Instead of: config = AutoConfig.from_pretrained("model-name")
   # Do:
   HIDDEN_DIM = 768
   NUM_LAYERS = 12
   NUM_HEADS = 12
   INTERMEDIATE_SIZE = 3072
   ```

3. **Wraps the model in the standard interface**:
   ```python
   class Model(nn.Module):
       def __init__(self):
           super().__init__()
           self.model = OriginalModelClass(hardcoded_config)

       def forward(self, *args):
           return self.model(*args)

   def get_inputs():
       # Return concrete tensors matching the model's expected inputs
       return [torch.randint(0, 32000, (1, 128)),  # input_ids
               torch.ones(1, 128, dtype=torch.long)]  # attention_mask

   def get_init_inputs():
       return []

   # Optional: for models with keyword-only arguments
   def get_input_kwargs():
       return {"attention_mask": torch.ones(1, 128, dtype=torch.long)}
   ```

4. **Loads pretrained weights (optional)**: If needed, load weights in `__init__` from a local checkpoint. The wrapper must not require network access at runtime.

### Step 0.3: Handle External Dependencies

- **Pure PyTorch ops** (`einops.rearrange`, etc.): Inline the equivalent torch code
- **Heavy frameworks** (`transformers`, `timm`): Either:
  - (a) Copy only the needed source files locally and adjust imports, OR
  - (b) Accept the pip dependency and document it in a `requirements.txt`
- **Custom CUDA kernels**: Replace with PyTorch-equivalent ops (mark as approximate if needed)

### Step 0.4: Verify the Wrapper (GATE)

This is the Phase 0 verification gate. **You MUST NOT proceed to Phase 1 until this passes.**

#### Stage A: Standalone Execution

Run the wrapper file. It must:
- Execute without errors: `python {output_dir}/level_3_model/{model_name}.py` → prints PASS
- Produce deterministic output shapes
- NOT require network access or filesystem access at runtime

#### Stage B: Numerical Equivalence Against Original

Create `{output_dir}/steps/step_0_preparation/verify_wrapper.py` that:

1. **Loads the original model using its native interface** (HF `transformers`, multi-file imports, config objects — whatever it needs). This is the **only** place in the pipeline where external dependencies are allowed.
2. **Loads the self-contained wrapper** using the standard `Model`/`get_inputs()` interface
3. **Transfers weights** from the original to the wrapper
   - For HuggingFace: `original = AutoModel.from_pretrained(...)` or `AutoModel.from_config(config)`, then map `state_dict` keys to the wrapper's `state_dict`
   - For multi-file repos: load the original modules, map weights
4. **Runs both on identical inputs** (3 trials, same seeds)
5. **Compares outputs** with the same tolerances as `verify_step.py`

```python
"""
Phase 0 Verification: Wrapper vs Original Model
Confirms the self-contained wrapper reproduces the original model's output.
"""
import torch
import sys
from pathlib import Path

# ---- Load original model (using native dependencies) ----
# This section IS allowed to use transformers, timm, etc.
from transformers import AutoModel, AutoConfig

config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
original = AutoModel.from_config(config)  # random init, same architecture
original.eval()

# ---- Load wrapper model (self-contained) ----
sys.path.insert(0, str(Path(__file__).parent.parent / "level_3_model"))
from smolvlm import Model, get_inputs, get_init_inputs

wrapper = Model(*get_init_inputs())
wrapper.eval()

# ---- Transfer weights: original → wrapper ----
orig_sd = original.state_dict()
wrap_sd = wrapper.state_dict()

mapped = 0
for key in wrap_sd:
    if key in orig_sd and orig_sd[key].shape == wrap_sd[key].shape:
        wrap_sd[key] = orig_sd[key].clone()
        mapped += 1
    # Also try with prefix adjustments, e.g.:
    # "model.encoder.layers.0.weight" ↔ "encoder.layers.0.weight"

wrapper.load_state_dict(wrap_sd)
print(f"Mapped {mapped}/{len(wrap_sd)} parameters")

# ---- Numerical comparison (3 trials) ----
num_trials = 3
max_diff_all = 0.0
all_pass = True

for trial in range(num_trials):
    torch.manual_seed(42 + trial)
    inputs = get_inputs()

    with torch.no_grad():
        orig_out = original(*inputs)
        wrap_out = wrapper(*inputs)

    # Handle HF model outputs (may be dataclass/tuple, not raw tensor)
    if hasattr(orig_out, "last_hidden_state"):
        orig_tensor = orig_out.last_hidden_state
    elif isinstance(orig_out, tuple):
        orig_tensor = orig_out[0]
    else:
        orig_tensor = orig_out

    if hasattr(wrap_out, "last_hidden_state"):
        wrap_tensor = wrap_out.last_hidden_state
    elif isinstance(wrap_out, tuple):
        wrap_tensor = wrap_out[0]
    else:
        wrap_tensor = wrap_out

    diff = (orig_tensor.float() - wrap_tensor.float()).abs().max().item()
    max_diff_all = max(max_diff_all, diff)
    matches = torch.allclose(orig_tensor.float(), wrap_tensor.float(),
                              rtol=1e-5, atol=1e-6)
    if not matches:
        all_pass = False
    print(f"Trial {trial}: max_diff={diff:.2e} {'PASS' if matches else 'FAIL'}")

print(f"\n{'PASS' if all_pass else 'FAIL'} (max_diff={max_diff_all:.2e})")
sys.exit(0 if all_pass else 1)
```

#### What This Catches

| Failure | Cause |
|---|---|
| Shape mismatch | Wrapper has wrong config values (hidden_dim, num_heads, etc.) |
| Large numerical diff | Missing operation (e.g., forgot a normalization layer during extraction) |
| Weight mapping gaps | Wrapper restructured the module tree, breaking state_dict key names |
| Output format mismatch | Wrapper returns raw tensor but original returns a ModelOutput dataclass |

#### If verification fails:
- Check weight mapping — print unmapped keys from both sides
- Check that the wrapper's `forward()` includes ALL operations (norms, activations, residuals)
- Check config values match exactly
- For HF models: ensure you handle the output format (`.last_hidden_state`, `.logits`, etc.)

#### PASS criteria:
- All 3 trials pass with `rtol=1e-5, atol=1e-6` (float32) or `rtol=1e-3, atol=1e-3` (bfloat16/float16)
- Weight mapping covers >95% of parameters (some buffers like position IDs may not need mapping)

Once Phase 0 passes, the wrapper file IS your input to Phase 1.

---

## Phase 1: Analysis

### Step 1.0: Resolve Configuration (config-driven models only)

If the model uses a config object (HuggingFace-style):
1. Pick a **concrete configuration** (specific model size/variant)
2. Hardcode ALL config values as constants or `__init__` parameters
3. Do NOT use `AutoConfig`, `from_pretrained()`, or any dynamic loading
4. Document the exact config used in `decomposition_analysis.md`

Example — for SmolVLM-256M, resolve all config values:
```python
HIDDEN_SIZE = 768
NUM_ATTENTION_HEADS = 12
NUM_HIDDEN_LAYERS = 12
INTERMEDIATE_SIZE = 3072
VOCAB_SIZE = 49152
```

> If your model already uses hardcoded dimensions (e.g., KernelBench models), skip to Step 1.1.

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

Classify by **semantic role**, not raw operation count. A single "operation" may internally use multiple primitives (e.g., RMSNorm uses pow, mean, rsqrt, multiply — but it's semantically one normalization).

**Classification guide:**

| Level | Criteria | Examples |
|-------|----------|----------|
| 0 (Kernel) | A single logical operation that stands alone — either it can't be fused with its neighbors (due to residual connections, data flow breaks), or it's an atomic op. May use multiple primitives internally. | `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, RMSNorm, `nn.Embedding`, Softmax, SiLU, GELU, MatMul |
| 1 (Fusion) | Adjacent operations that CAN be fused (no data flow breaks between them). They form a straight-line computation without residual/skip connections interrupting. | Linear+SiLU (SwiGLU gate), QKV proj+reshape+attention scores, Conv+BN+ReLU |
| 2 (Layer) | A logical building block, often repeated. Contains fusions and/or kernels with residual connections between them. | TransformerBlock, ResNetBlock, EncoderLayer, DecoderLayer |
| 3 (Model) | Complete model with multiple layers/stages | Full Transformer, VisionEncoder, LLM |

**Key distinction between kernel and fusion:**
- **Fusion (L1)**: Groups operations that CAN be fused together — i.e., executed in a single kernel launch without intermediate memory reads/writes. The operations are adjacent in the data flow with no branching (residual connections, skip connections) between them. Examples: Linear+SiLU (projection feeds directly into activation), Conv+BN+ReLU, QKV projection+reshape.
- **Kernel (L0)**: A single logical operation that either stands alone (can't be fused with its neighbors due to residual connections or data flow breaks) or is already an atomic operation. Examples: RMSNorm (typically separated from adjacent ops by a residual add), LayerNorm, Softmax, standalone Linear projections.

**Classification process:**
1. First, identify groups of adjacent operations that can be fused (no data flow breaks between them) → these become **fusions (L1)**
2. Any remaining operations that can't be grouped with neighbors → these become **kernels (L0)**

**Example — TransformerBlock decomposition:**
```
residual = x
x = self.rms_norm(x)          # L0 kernel (can't fuse with residual add above)
x = self.attention(x)          # L1 fusion (QKV proj + matmul + softmax + matmul + out proj)
x = x + residual               # data plumbing (residual add breaks fusion opportunity)
residual = x
x = self.rms_norm_2(x)         # L0 kernel (isolated by residual adds)
x = self.mlp(x)                # L1 fusion (Linear + SiLU + Linear)
x = x + residual               # data plumbing
```

### Step 1.3: Identify Decomposition Targets

For the CURRENT level, identify children by their **semantic role**. Prefer next-level-down, but classify each child at its natural level:

| Current Level | Primary Children | May Also Include |
|--------------|------------------|------------------|
| Model (L3) | Layers (L2): repeated blocks, major sub-modules | Kernels (L0): standalone embeddings, projections |
| Layer (L2) | Fusions (L1): attention block, MLP block | Kernels (L0): norms, activations that stand alone |
| Fusion (L1) | Kernels (L0): individual operations | — (L0 is the leaf level) |

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

**All parameters must belong to child submodules imported from child files.** No standalone `nn.Linear`, `nn.Conv2d`, `nn.Embedding`, etc. constructed directly in the refactored model's `__init__`. Every `nn.Module` child must be an instance of a class imported from a child component file. Simple operations (embedding lookups, linear projections) that exist at the top level of a model should be extracted as kernel-level child files.

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
1. Model (L3) → children (primarily L2 layers, may include L1 fusions or L0 kernels) — verify step
2. For each non-kernel child: decompose further — verify each step
3. Continue until all leaf nodes are at kernel level (L0)

Each decomposition step may produce a **mix of levels** among its children. For example, decomposing a Layer (L2) might yield 2 Fusions (L1) + 1 Kernel (L0). Only the non-kernel children need further decomposition.

### Multi-Modal / Composite Models

For models with distinct sub-systems (e.g., vision encoder + LLM + connector like SmolVLA):

1. **Level 3 decomposition should first separate the major sub-systems** — vision encoder, language model, projection/connector layers each become Level 2 components
2. **Each sub-system becomes its own Level 2 decomposition target** — decompose independently with its own verification steps
3. **Tokenizers and processors are NOT part of the computation graph** — exclude them entirely; they produce the tensors that `get_inputs()` returns
4. **Weight tying between sub-systems** (e.g., shared embeddings) must be explicitly documented and handled in `weight_map.json`
5. **Multiple input modalities** — the parent's `get_inputs()` returns all modalities; each child receives only the inputs it needs. Document which parent inputs flow to which children in the architecture analysis.

### Kernel Deduplication

Many models reuse the same operation with the same shapes multiple times (e.g., a transformer with 12 identical layers each containing the same Linear projection). **Do NOT create duplicate kernel files.**

**Kernel identity (= unique workload)** is defined by: `(operation, parameter_shapes, input_dtype, output_dtype)`. If two kernels match on ALL of these, they are the **same workload** and share a single file.

**The file name encodes the workload identity**, while the module uses `get_init_inputs()` to pass workload-specific dimensions:
```python
# level_0_kernel/linear_768x3072_fp32.py
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.linear(x)

def get_init_inputs():
    return [768, 3072]  # workload-specific dimensions

def get_inputs():
    return [torch.randn(1, 128, 768)]  # input matching this workload
```

**Exception — shape-independent elementwise ops** (SiLU, GELU, ReLU, etc.) don't have parameter shapes and work on any input shape. These get ONE file each (e.g., `silu_fp32.py`) regardless of where they're used.

**Rules:**
1. **One file per unique workload** — e.g., one `linear_768x3072_fp32.py` even if 12 transformer layers each have this same projection
2. **Multiple instances import from the same file** — in `refactored.py`, create separate instances with different names:
   ```python
   from children.linear_768x3072_fp32 import Model as UpProj0
   from children.linear_768x3072_fp32 import Model as UpProj1
   # Both are separate nn.Module instances (separate weights),
   # but defined by the same kernel file
   ```
3. **Weight transfer maps different weights to each instance** — instance 0 gets layer 0's weights, instance 1 gets layer 1's weights, etc.
4. **`decomposition_tree.json` tracks instances** — multiple nodes can reference the same `path` but with different `id`s. Add an `instance_count` field to kernel nodes to document how many times this kernel appears in the model.
5. **Naming: no `_2`, `_3` suffixes** — do NOT create `relu_fp32.py`, `relu_fp32_2.py`, `relu_fp32_3.py`. Just create `relu_fp32.py` once.

**When are two kernels NOT duplicates (= different workloads)?**
- Different parameter shapes: `linear_768x3072_fp32.py` vs `linear_3072x768_fp32.py`
- Different dtypes: `linear_768x3072_fp32.py` vs `linear_768x3072_bf16.py`
- Different parameter configurations: `conv2d_3x768_16x16_fp32.py` vs `conv2d_768x768_3x3_fp32.py`
- Different operations: `silu_fp32.py` vs `gelu_fp32.py`

**Naming format:** `{operation}_{shape_signature}_{dtype}.py` — see OUTPUT_SCHEMA.md for full specification.

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

### Multi-Input Models

If the original model takes multiple named inputs (e.g., `input_ids`, `attention_mask`, `pixel_values`), `get_inputs()` should return them as a list in positional order matching the `forward()` signature:

```python
def get_inputs():
    return [
        torch.randint(0, 32000, (1, 128)),      # input_ids
        torch.ones(1, 128, dtype=torch.long),    # attention_mask
        torch.randn(1, 3, 224, 224),             # pixel_values
    ]
```

For models with keyword-only arguments, also define `get_input_kwargs()`:

```python
def get_input_kwargs():
    """Return keyword arguments for forward()."""
    return {"attention_mask": torch.ones(1, 128, dtype=torch.long)}
```

When decomposing, each child component receives only the inputs it needs. Document which parent inputs flow to which children in the architecture analysis.

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

### Step 3.2: Coverage Summary (REQUIRED)

Run full coverage analysis comparing original model ops against **leaf-level (L0 kernel) components only**. Do NOT count ops from L1/L2 components — they contain the same ops nested inside them and would inflate the count.

**You MUST produce `coverage_summary.json` — it is a required output artifact.**

```bash
python scripts/extract_ops.py \
    --model {original_model.py} \
    --decomp-dir {output_dir}/ \
    --output {output_dir}/verification/coverage_summary.json
```

> **Note**: `--decomp-dir` scans only `level_0_kernel/` to avoid double/triple-counting ops that appear at multiple hierarchy levels. If you need per-step coverage, use `--children` with explicit file paths instead.

The coverage summary must show:
- Total unique op types in original model vs. L0 kernels
- 1:1 matching of op types (each original op type has a corresponding kernel file)
- Instance counts (how many times each kernel is used across the model)
- Any missing or extra operations

### Step 3.3: Step Pipeline Summary (optional, for CI/human review)

Re-validate all steps and aggregate results:
```bash
python scripts/run_step_pipeline.py {output_dir}/
```
This is redundant if the agent already ran `verify_step.py` at each step in Phase 2. It exists for post-hoc validation by humans or CI systems.

---

## Decision & Difficulty Log

As you decompose, maintain a structured log at `{output_dir}/decomposition_log.json`. Record decisions, difficulties, and deviations as you encounter them — this provides a reviewable audit trail.

**Format:**
```json
{
  "entries": [
    {
      "step": "step_1_model_to_layers",
      "type": "decision",
      "title": "Classified RMSNorm as kernel, not fusion",
      "description": "RMSNorm uses pow/mean/rsqrt/multiply internally, but semantically it's a single normalization operation. Classified as L0 kernel per semantic classification guide.",
      "timestamp": "2026-02-27T03:15:00Z"
    },
    {
      "step": "step_4_vlm_expert_transformer",
      "type": "difficulty",
      "title": "Anti-cheat flagged torch.where in masking",
      "description": "The attention masking pattern uses torch.where(mask, attn, -inf) which is in the disallowed list. Replaced with arithmetic equivalent: attn * mask.float() + (-1e9) * (~mask).float()",
      "resolution": "Used arithmetic masking as data plumbing",
      "timestamp": "2026-02-27T05:30:00Z"
    },
    {
      "step": "phase_0",
      "type": "deviation",
      "title": "Added VisionEncoderBlock wrapper class",
      "description": "HuggingFace uses encoder.layers.X naming but wrapper uses layers.X. Added intermediate class to bridge state_dict key names.",
      "resolution": "Weight mapping now works correctly",
      "timestamp": "2026-02-27T01:00:00Z"
    }
  ]
}
```

**Entry types:**
- **decision**: Classification choices (kernel vs fusion), architectural trade-offs, naming decisions
- **difficulty**: Issues encountered and how they were resolved (weight mismatches, anti-cheat workarounds, numerical precision issues)
- **deviation**: Any place you deviated from the standard instructions and why

**When to log:** After each verification step, and whenever you make a non-obvious choice. If you encounter a problem that takes more than one attempt to fix, log it.

---

## Phase 4: Output Structure

```
{output_dir}/
├── level_3_model/
│   └── {model_name}.py
├── level_2_layer/
│   ├── transformer_block.py
│   └── ...
├── level_1_fusion/
│   ├── attention_block.py
│   ├── mlp_block.py
│   └── ...
├── level_0_kernel/                          # Unique workloads with shape-based names
│   ├── embedding_50000x768_fp32.py          # Hardcoded: vocab=50000, dim=768
│   ├── linear_768x3072_fp32.py              # Hardcoded: in=768, out=3072
│   ├── linear_3072x768_fp32.py              # Different workload from above
│   ├── layer_norm_768_fp32.py               # Hardcoded: normalized_shape=768
│   ├── softmax_fp32.py
│   ├── gelu_fp32.py
│   └── ...
├── steps/
│   ├── step_0_preparation/              # Phase 0 (multi-file models only)
│   │   ├── verify_wrapper.py
│   │   └── verification_result.json
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
│   └── coverage_summary.json               # REQUIRED: from extract_ops.py
├── decomposition_tree.json
├── decomposition_analysis.md
└── decomposition_log.json                   # Decision/difficulty log
```

---

## Constraints

### You MUST NOT:
1. Copy the entire model as a single "kernel"
2. Force components into the wrong abstraction level (e.g., wrapping a single-op kernel as a fake "fusion" just to maintain strict level ordering)
3. Combine unrelated operations into fake "fusions"
4. Invent operations not in the original
5. Generate files that don't execute
6. **Modify, reduce, or simplify the dimensions declared in the original model** — Do NOT shrink batch size, channel counts, hidden dimensions, sequence lengths, or any other parameter for "faster testing". Use the EXACT values from the original model's declarations and `get_inputs()`/`get_init_inputs()` functions
7. **Put compute ops in refactored code** — refactored forward() must ONLY call child modules + data plumbing. No `F.relu`, `torch.matmul`, `nn.Linear`, etc.
8. **Construct raw nn.Modules in refactored code** — every `nn.Module` child in `RefactoredModel.__init__()` must be imported from a child component file. No `nn.Linear(...)`, `nn.Embedding(...)`, `nn.Conv2d(...)`, etc. directly
9. **Proceed to the next level before the current step's verify_step.py PASSES**
10. **Use generic kernel file names** — kernel files must use shape-based naming: `{operation}_{shape_signature}_{dtype}.py`

### You MUST:
1. Trace complete data flow with shapes at each step
2. Preserve all dtypes (float32, bfloat16, etc.)
3. **Preserve the EXACT dimensions from the original model** — Use the same batch size, channel counts, hidden dimensions, kernel sizes, sequence lengths, and all other parameters as declared in the original model's `__init__()`, `get_inputs()`, and `get_init_inputs()`. Never reduce or simplify dimensions
4. **Decompose one level at a time** with verification between each step
5. **Create refactored.py at each step** and run `verify_step.py` to confirm numerical equivalence
6. Create final composition_test.py that PASSES
7. Run `extract_ops.py` to produce `coverage_summary.json`
8. Verify every component executes with correct shapes
9. Document reasoning for each decomposition decision in `decomposition_log.json`
10. **Use semantic classification** for abstraction levels — single logical operations (RMSNorm, RoPE, Softmax) are kernel-level even if their implementation uses multiple primitives

---

## Execution Checklist

Before submitting, verify:

- [ ] **Each decomposition step** has a refactored.py with ONLY child module calls (no raw `nn.Linear`, `nn.Embedding`, etc.)
- [ ] **Each step's verify_step.py PASSES** (numerical equivalence confirmed, anti-cheat passed)
- [ ] All component files execute without error (print "PASS")
- [ ] `composition_test.py` PASSES (final composed output matches original)
- [ ] `decomposition_tree.json` is complete
- [ ] Every leaf node is at kernel level (L0)
- [ ] Abstraction hierarchy is respected (children always at a lower level than parent; prefer next level down)
- [ ] **`coverage_summary.json` exists** in `verification/` directory (from `extract_ops.py`)
- [ ] Coverage analysis shows no missing operations
- [ ] **Kernel files use shape-based naming**: `{operation}_{shape_signature}_{dtype}.py` with hardcoded dimensions
- [ ] **Kernel classification is semantic**: single logical operations (RMSNorm, RoPE, etc.) are L0, not L1
- [ ] `decomposition_log.json` records key decisions, difficulties, and deviations
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
