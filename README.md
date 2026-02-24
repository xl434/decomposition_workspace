# Hierarchical Decomposition Workspace

Decompose PyTorch models into hierarchical unit tests using an LLM agent. Each model is broken down level by level — Model → Layer → Fusion → Kernel — with **verified correctness at every step**.

## How It Works

```
Original PyTorch Model
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  Step-by-step decomposition with gates:      │
  │                                              │
  │  Model → Layers     ── verify_step.py PASS ──│─► proceed
  │  Layer → Fusions    ── verify_step.py PASS ──│─► proceed
  │  Fusion → Kernels   ── verify_step.py PASS ──│─► proceed
  └─────────────────────────────────────────────┘
        │
        ▼
  Verified kernel-level unit tests
  + end-to-end composition test
```

At each step, the agent:
1. Extracts child components at the next level
2. Rewrites the parent as a **refactored model** that calls only child modules
3. Runs `verify_step.py` to prove the refactored version produces identical output (shared weights, same input)
4. Only proceeds after PASS

---

## Quick Start

### 1. Pick a model

```bash
# Simple (good for first try)
examples/simple_transformer.py

# KernelBench dataset (251 models across 3 difficulty levels)
data/kernelbench/level1/   # 100 atomic ops (already kernel-level, no decomposition needed)
data/kernelbench/level2/   # 100 fused ops (Fusion → Kernel)
data/kernelbench/level3/   # 51 full models (Model → Layer → Fusion → Kernel)
```

### 2. Copy the agent prompt

Open `AGENT_PROMPT_TEMPLATE.md`, copy the **Standard Prompt**, and fill in the `[EDIT]` sections with your model path and output directory. Then paste it into your LLM agent (Claude, etc.).

**Minimal prompt (copy, edit the two [EDIT] lines, paste):**
```
I need you to decompose a PyTorch model into hierarchical unit tests.

Read and follow: decomposition_workspace/prompts/MAIN_PROMPT.md
Read verification rules: decomposition_workspace/prompts/VERIFICATION_GUIDE.md
Read output schema: decomposition_workspace/OUTPUT_SCHEMA.md

[EDIT] Model: decomposition_workspace/data/kernelbench/level3/11_VGG16.py
[EDIT] Output: decomposition_workspace/output/level3/11_VGG16/

Decompose step by step. Run verify_step.py at each level. Do not proceed until PASS.
```

See `AGENT_PROMPT_TEMPLATE.md` for the full prompt with detailed round-by-round instructions, batch decomposition examples, and tips.

### 3. Verify the results

```bash
# Check all step-by-step verification results
python scripts/run_step_pipeline.py output/level3/11_VGG16/

# Run all component self-tests
python scripts/run_all_tests.py output/level3/11_VGG16/

# Run the end-to-end composition test
python output/level3/11_VGG16/verification/composition_test.py

# Or batch-test everything in output/
python scripts/batch_test.py --verbose
```

---

## Verification System

Two layers of verification ensure decomposition correctness:

### Layer 1: Step Verification (primary, at each level)

Every time the agent decomposes a component, it creates a `refactored.py` that replaces inline computation with calls to child modules. `verify_step.py` then:

1. **Anti-cheat check** — scans `refactored.py` to ensure `forward()` only calls child modules + data plumbing (no `F.relu`, `torch.matmul`, `nn.Linear`, etc.)
2. **Weight transfer** — maps parameters from original to refactored model
3. **Numerical comparison** — runs both models on same inputs (3 trials), asserts outputs match

```bash
python scripts/verify_step.py \
    --original path/to/parent.py \
    --refactored path/to/refactored.py \
    --output path/to/verification_result.json
```

### Layer 2: End-to-End Composition Test (final double-check)

After all steps pass, `composition_test.py` builds a single model from only L0 kernel components and verifies it matches the original. This is an independent proof that doesn't rely on step results.

### Coverage Analysis

`extract_ops.py` uses torch.fx (with torch.compile and forward-hook fallbacks) to extract all operations from the original model and compare against decomposed components:

```bash
python scripts/extract_ops.py \
    --model path/to/original.py \
    --decomp-dir output/level3/11_VGG16/ \
    --output output/level3/11_VGG16/verification/coverage_summary.json
```

---

## Directory Structure

```
decomposition_workspace/
├── prompts/                         # Agent instructions
│   ├── MAIN_PROMPT.md              # Full decomposition methodology
│   └── VERIFICATION_GUIDE.md       # Verification rules & edge cases
├── scripts/                         # Verification & testing tools
│   ├── verify_step.py              # Per-step verification (agent calls this)
│   ├── extract_ops.py              # Op coverage analysis (torch.fx/compile/hooks)
│   ├── run_step_pipeline.py        # Validate all steps in a decomposition
│   ├── run_all_tests.py            # Run all component self-tests
│   └── batch_test.py              # Batch-test all decompositions
├── verification/                    # Templates
│   ├── composition_template.py     # End-to-end composition test template
│   └── step_refactored_template.py # Refactored code template
├── data/                            # Input models
│   └── kernelbench/                # KernelBench dataset (251 models)
│       ├── level1/                 # 100 atomic ops
│       ├── level2/                 # 100 fused ops
│       └── level3/                 # 51 full models
├── examples/                        # Self-contained example models
│   ├── simple_transformer.py
│   ├── attention_block.py
│   └── mlp_block.py
├── output/                          # Decomposition outputs
├── AGENT_PROMPT_TEMPLATE.md        # Copy-paste prompts for agents
├── OUTPUT_SCHEMA.md                # Output directory & file format rules
└── README.md
```

### Output Structure (per decomposition)

```
output/level3/model_name/
├── level_3_model/*.py              # Model-level components
├── level_2_layer/*.py              # Layer-level components
├── level_1_fusion/*.py             # Fusion-level components
├── level_0_kernel/*.py             # Kernel-level components (leaf nodes)
├── steps/                          # Intermediate verification results
│   ├── step_1_model_to_layers/
│   │   ├── original.py             # Copy of parent
│   │   ├── refactored.py           # Parent rewritten with child calls
│   │   ├── children/               # Child component files
│   │   ├── verification_result.json
│   │   └── coverage_report.json
│   ├── step_2_layer_to_fusions/
│   └── step_3_fusion_to_kernels/
├── verification/
│   ├── composition_test.py         # End-to-end composition proof
│   ├── step_verification_summary.json
│   └── coverage_summary.json
├── decomposition_tree.json
└── decomposition_analysis.md
```

---

## Scripts Reference

| Script | What it does | Who runs it |
|--------|-------------|-------------|
| `verify_step.py` | Compares original vs refactored model (anti-cheat + numerical) | Agent (at each step) |
| `extract_ops.py` | Extracts ops via torch.fx and checks coverage | Agent (final) or user |
| `run_step_pipeline.py` | Re-validates all steps in a decomposition | User (post-hoc check) |
| `run_all_tests.py` | Runs every component's self-test | User |
| `batch_test.py` | Tests all decompositions in output/ | User |

---

## Abstraction Levels

| Level | Name | # Ops | Examples | Decomposes to |
|-------|------|-------|---------|---------------|
| 3 | Model | 15+ or loops | GPT, ViT, VGG16, LLaMA | Layers |
| 2 | Layer | 6-15 | TransformerBlock, ResNetBlock | Fusions |
| 1 | Fusion | 2-5 | Conv+BN+ReLU, QKV projection | Kernels |
| 0 | Kernel | 1 | Linear, ReLU, Softmax, Conv2d | (leaf) |

---

## Key Concepts

### The Refactored Code Pattern

At each decomposition step, the agent creates a "refactored" version of the parent that replaces inline computation with calls to child modules:

```python
# Original: inline computation
class Model(nn.Module):
    def forward(self, x):
        x = self.conv(x)          # compute
        x = F.relu(x)             # compute
        x = self.pool(x)          # compute
        return x

# Refactored: child module calls only
class RefactoredModel(nn.Module):
    def forward(self, x):
        x = self.conv_relu(x)     # child call
        x = self.pool(x)          # child call
        return x
```

`verify_step.py` proves these produce identical outputs with shared weights.

### Anti-Cheat Rules

Refactored `forward()` may only contain:
- Child module calls (`self.child(x)`)
- Data plumbing (`x + residual`, `torch.cat(...)`)
- Shape ops (`x.reshape(...)`, `x.permute(...)`)

It must NOT contain compute ops (`F.relu`, `torch.matmul`, `nn.Linear`, etc.).

---