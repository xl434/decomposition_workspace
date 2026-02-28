# Agent Prompt Template

Copy the prompt below, edit the single `[EDIT]` block at the top to specify your model(s) and output directory, then paste into your LLM agent.

---

## Prompt (copy this)

```
I need you to hierarchically decompose a PyTorch model into unit tests,
one level at a time with verification at each step.

## ──── [EDIT THIS SECTION] ────────────────────────────────────────────

Input type:           single_file
Model to decompose:   decomposition_workspace/data/kernelbench/level3/11_VGG16.py
Output directory:     decomposition_workspace/output/level3/11_VGG16/

## ──────────────────────────────────────────────────────────────────────

## Instructions

Read and follow these files:
- decomposition_workspace/prompts/MAIN_PROMPT.md (full methodology)
- decomposition_workspace/prompts/VERIFICATION_GUIDE.md (verification rules)
- decomposition_workspace/OUTPUT_SCHEMA.md (output structure)

## Protocol: Step-by-Step with Verification Gates

Do NOT decompose all levels at once. For each level transition:

1. Decompose the current component into next-level children
2. Create a refactored.py that calls ONLY child modules (no inline computation)
3. Run verify_step.py — MUST PASS before proceeding to the next level
4. Repeat until all components reach kernel level (L0)

### Round 0: Model Preparation (huggingface / repo input types only)
Skip this round if input type is single_file with no externel file dependencies.
- Follow Phase 0 in MAIN_PROMPT.md
- Step 0.1: Explore the entry file and trace ALL imports recursively to discover
  dependencies, config values, forward signature, and input shapes
- Step 0.2-0.3: Create a self-contained wrapper with hardcoded config
- Step 0.4: Create steps/step_0_preparation/verify_wrapper.py
- Run verify_wrapper.py — MUST PASS before Round 1

### Round 1: Model -> Layers
- Create level_2_layer/*.py files
- Create steps/step_1_model_to_layers/refactored.py
- Run: python scripts/verify_step.py --original <parent> --refactored <refactored>
- PASS required before Round 2

### Round 2: Each Layer -> Fusions
- Create level_1_fusion/*.py files
- Create steps/step_2_{layer}_to_fusions/refactored.py for each layer
- Run verify_step.py for each — ALL must PASS before Round 3

### Round 3: Each Fusion -> Kernels
- Create level_0_kernel/*.py files
- Create steps/step_3_{fusion}_to_kernels/refactored.py for each fusion
- Run verify_step.py for each — ALL must PASS

### Final
- Run: python scripts/extract_ops.py --model <original> --decomp-dir <output_dir>
- Create and run verification/composition_test.py

## Success Criteria

- [ ] Phase 0 verify_wrapper.py PASSES (if applicable)
- [ ] Every verify_step.py PASSES (anti-cheat + numerical equivalence)
- [ ] All component files execute without error (print "PASS")
- [ ] Final composition_test.py PASSES
- [ ] All dimensions match the original model EXACTLY
- [ ] Every leaf node is kernel level (single operation)
- [ ] decomposition_tree.json is complete

Begin decomposition:
```

---

## Examples of the [EDIT] section

**Single KernelBench model (single_file):**
```
Input type:           single_file
Model to decompose:   decomposition_workspace/data/kernelbench/level3/11_VGG16.py
Output directory:     decomposition_workspace/output/level3/11_VGG16/
```

**Multiple models (agent works through them one by one):**
```
Input type:           single_file
Models to decompose (under data/kernelbench/level3/):
  - 11_VGG16.py
  - 14_DenseNet121.py
  - 18_SqueezeNet.py
Output directory:     decomposition_workspace/output/level3/{model_name}/
```

**Custom model (single_file):**
```
Input type:           single_file
Model to decompose:   decomposition_workspace/my_models/custom_unet.py
Output directory:     decomposition_workspace/output/level3/custom_unet/
```

**HuggingFace model:**
```
Input type:           huggingface
Model ID or repo:     HuggingFaceTB/SmolVLM-256M-Instruct
Model class:          SmolVLMForConditionalGeneration
Output directory:     decomposition_workspace/output/level3/smolvlm/
```

**Multi-file repo:**
```
Input type:           repo
Repo path:            decomposition_workspace/data/smolvla/
Entry file:           modeling_smolvla.py
Entry class:          SmolVLAForConditionalGeneration
Output directory:     decomposition_workspace/output/level3/smolvla/
```

> For `huggingface` and `repo` input types, the agent will automatically explore
> the source code to discover dependencies, config values, forward signature,
> and input shapes during Phase 0 (see MAIN_PROMPT.md Step 0.1).

---

## Available Models

### KernelBench Dataset (`data/kernelbench/`)
| Level | Count | Description | Example |
|-------|-------|-------------|---------|
| level1 | 100 | Atomic ops (already kernel-level) | `19_ReLU.py` |
| level2 | 100 | Fused ops (2-5 operations) | `conv_bn_relu.py` |
| level3 | 51 | Full models | `11_VGG16.py`, `gpt_oss.py` |
