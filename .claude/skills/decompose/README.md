# Decompose Skill for Claude Code

Hierarchically decompose a PyTorch model into verified 4-level components: Model (L3) → Layer (L2) → Fusion (L1) → Kernel (L0).

## What It Does

Given a PyTorch model, this skill breaks it down into standalone, numerically verified components at four abstraction levels:

| Level | Name | Examples |
|-------|------|----------|
| 3 | Model | VGG16, Vision Transformer, VLM |
| 2 | Layer | TransformerBlock, ResNetBlock |
| 1 | Fusion | Conv+BN+ReLU, SwiGLU, QKV projection |
| 0 | Kernel | Conv2d, Linear, ReLU, LayerNorm |

Each decomposition step is verified for numerical equivalence and anti-cheat compliance before proceeding to the next level.

## Installation

This skill is available in the [decomposition_workspace](https://github.com/liang/decomposition_workspace) repo under `.claude/skills/decompose/`.

To use it in your own project, copy the skill directory:

```bash
cp -r .claude/skills/decompose/ /path/to/your/project/.claude/skills/decompose/
```

Or clone it into your personal skills directory for global access:

```bash
cp -r .claude/skills/decompose/ ~/.claude/skills/decompose/
```

## Usage

```
/decompose data/kernelbench/level3/11_VGG16.py
/decompose huggingface:HuggingFaceTB/SmolVLM-256M-Instruct
/decompose repo:data/lerobot/src/lerobot/policies/smolvla/
```

### Input Types

- **Single file**: Path to a `.py` file containing a PyTorch model
- **HuggingFace model**: `huggingface:<model-id>` to download and decompose a HuggingFace model
- **Repo model**: `repo:<directory>` to decompose a model from local source code

### Output

The skill produces a structured output directory:

```
output/{level}/{model_name}/
├── level_2_layer/          # Layer components
├── level_1_fusion/         # Fusion components
├── level_0_kernel/         # Kernel components (leaf nodes)
├── steps/                  # Verification steps with refactored code
├── verification/           # Composition tests
├── decomposition_tree.json # Full hierarchy
├── decomposition_log.json  # Decisions and difficulties
└── coverage_summary.json   # Op coverage analysis
```

## Skill Contents

```
.claude/skills/decompose/
├── SKILL.md                              # Skill entry point
├── references/
│   ├── MAIN_PROMPT.md                    # Full decomposition methodology
│   ├── VERIFICATION_GUIDE.md             # Verification rules and tolerances
│   └── OUTPUT_SCHEMA.md                  # Output structure and naming conventions
└── scripts/
    ├── verify_step.py                    # Per-step anti-cheat + numerical verification
    ├── extract_ops.py                    # Coverage analysis (original vs decomposed ops)
    ├── run_step_pipeline.py              # Post-hoc validation of all steps
    ├── run_all_tests.py                  # Batch self-test runner
    ├── batch_test.py                     # Batch test across multiple decompositions
    ├── composition_template.py           # End-to-end composition test template
    └── step_refactored_template.py       # Refactored code template
```

## Requirements

- Python 3.10+
- PyTorch
- CUDA-capable GPU (recommended for larger models)
- Claude Code CLI

## How It Works

The skill follows a strict step-by-step protocol with verification gates:

1. **Round 0** (HuggingFace/repo models only): Trace imports, create self-contained wrapper
2. **Round 1**: Decompose model into layer components, verify equivalence
3. **Round 2**: Decompose each layer into fusion components, verify each
4. **Round 3**: Decompose each fusion into kernel components, verify each
5. **Final**: Run coverage analysis, composition test, generate reports
