---
name: decompose
description: Hierarchically decompose a PyTorch model into verified 4-level components (model, layer, fusion, kernel)
argument-hint: [model-path]
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Bash, Grep, Glob, Agent
---

Hierarchically decompose the PyTorch model at `$ARGUMENTS` into standalone,
verified components across 4 abstraction levels.

## Setup

The skill directory is `${CLAUDE_SKILL_DIR}`. It contains:
- `references/MAIN_PROMPT.md` — full methodology
- `references/VERIFICATION_GUIDE.md` — verification rules
- `references/OUTPUT_SCHEMA.md` — output structure and naming conventions
- `scripts/verify_step.py` — per-step verification (anti-cheat + numerical)
- `scripts/extract_ops.py` — coverage analysis (original ops vs decomposed kernels)
- `scripts/run_step_pipeline.py` — post-hoc validation of all steps
- `scripts/run_all_tests.py` — batch self-test runner for all components
- `scripts/batch_test.py` — batch test across multiple decompositions
- `scripts/composition_template.py` — template for end-to-end composition test
- `scripts/step_refactored_template.py` — template for refactored code

Copy `${CLAUDE_SKILL_DIR}/scripts/` into the workspace root as `scripts/` if it
does not already exist, so that verification commands work with standard paths.

## Argument Parsing

Parse `$ARGUMENTS` to determine the input type and paths:

- **Single file** (default): `$0` is a `.py` file path (e.g., `data/kernelbench/level3/11_VGG16.py`)
  - Input type: `single_file`
  - Output dir: `output/{level}/{model_name}/` (infer level and name from path)
- **HuggingFace model**: `$0` starts with `huggingface:` (e.g., `huggingface:HuggingFaceTB/SmolVLM-256M-Instruct`)
  - Input type: `huggingface`
  - Output dir: `output/level3/{model_name}/`
- **Repo model**: `$0` starts with `repo:` (e.g., `repo:data/lerobot/src/lerobot/policies/smolvla/`)
  - Input type: `repo`
  - Output dir: `output/level3/{model_name}/`

If `$1` is provided, use it as the output directory override.

## Required Reading

Before starting, read these files for full methodology, verification rules, and output structure:

1. [Full methodology](${CLAUDE_SKILL_DIR}/references/MAIN_PROMPT.md) — abstraction levels, phases, decomposition strategy, component file templates, anti-cheat rules, kernel deduplication
2. [Verification guide](${CLAUDE_SKILL_DIR}/references/VERIFICATION_GUIDE.md) — verification steps, tolerance guidelines, common failure modes, debugging approaches
3. [Output schema](${CLAUDE_SKILL_DIR}/references/OUTPUT_SCHEMA.md) — directory structure, naming conventions, kernel identity rules, decomposition tree format

## Protocol: Step-by-Step with Verification Gates

Do NOT decompose all levels at once. For each level transition:

1. Decompose the current component into next-level children
2. Create a `refactored.py` that calls ONLY child modules (no inline computation)
3. Run `verify_step.py` — MUST PASS before proceeding to the next level
4. Repeat until all components reach kernel level (L0)

### Round 0: Model Preparation (huggingface / repo input types only)

Skip this round if input type is `single_file` with no external file dependencies.

- Follow Phase 0 in MAIN_PROMPT.md
- Step 0.1: Explore the entry file and trace ALL imports recursively to discover dependencies, config values, forward signature, and input shapes
- Step 0.2-0.3: Create a self-contained wrapper with hardcoded config
- Step 0.4: Create `steps/step_0_preparation/verify_wrapper.py`
- Run `verify_wrapper.py` — MUST PASS before Round 1

### Round 1: Model → Layers

- Create `level_2_layer/*.py` files
- Create `steps/step_1_model_to_layers/refactored.py`
- Run: `python scripts/verify_step.py --original <parent> --refactored <refactored>`
- PASS required before Round 2

### Round 2: Each Layer → Fusions

- Create `level_1_fusion/*.py` files
- Create `steps/step_2_{layer}_to_fusions/refactored.py` for each layer
- Run `verify_step.py` for each — ALL must PASS before Round 3

### Round 3: Each Fusion → Kernels

- Create `level_0_kernel/*.py` files
- Create `steps/step_3_{fusion}_to_kernels/refactored.py` for each fusion
- Run `verify_step.py` for each — ALL must PASS

### Final

- Run: `python scripts/extract_ops.py --model <original> --decomp-dir <output_dir>`
- Create and run `verification/composition_test.py` (use `${CLAUDE_SKILL_DIR}/scripts/composition_template.py` as reference)
- Create `decomposition_tree.json`, `decomposition_log.json`, `decomposition_analysis.md`

## Abstraction Levels (Quick Reference)

| Level | Name | Description | Examples |
|-------|------|-------------|----------|
| 3 | Model | Complete models with multiple layers/stages | Transformer, VGG16, VLM |
| 2 | Layer | Higher-level building blocks, often repeated | TransformerBlock, ResNetBlock |
| 1 | Fusion | 2-5 adjacent ops commonly fused together | Conv+BN+ReLU, SwiGLU, QKV projection |
| 0 | Kernel | Single atomic operations | Conv2d, Linear, ReLU, LayerNorm, Softmax |

Key: Classify semantically. RMSNorm is one kernel (L0) even though it uses pow/mean/rsqrt internally. A standalone `nn.Embedding` inside a model is a kernel (L0), not a layer.

## Success Criteria

- [ ] Phase 0 `verify_wrapper.py` PASSES (if applicable)
- [ ] Every `verify_step.py` PASSES (anti-cheat + numerical equivalence)
- [ ] All component files execute without error (print "PASS")
- [ ] Final `composition_test.py` PASSES
- [ ] All dimensions match the original model EXACTLY
- [ ] Every leaf node is kernel level (L0, single operation)
- [ ] `decomposition_tree.json` is complete
- [ ] `coverage_summary.json` exists
- [ ] Kernel files use shape-based naming (e.g., `linear_768x3072_fp32.py`)
- [ ] `decomposition_log.json` records decisions and difficulties

Begin decomposition now.
