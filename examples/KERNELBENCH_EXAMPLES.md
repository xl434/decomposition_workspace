# KernelBench Examples

This workspace can decompose any model from the KernelBench dataset.

## Location

KernelBench is located at: `../KernelBench/KernelBench/`

## Available Levels

### Level 1 - Kernel Operations (Simple)
```
../KernelBench/KernelBench/level1/
```
- Simple operations: convolutions, linear layers, activations
- Already at kernel level - no decomposition needed
- Good for verifying kernel-level test generation

### Level 2 - Fused Operations
```
../KernelBench/KernelBench/level2/
```
- Fused operations: Conv+BN+ReLU, etc.
- Decompose to Level 0 kernels
- Good for testing fusion→kernel decomposition

### Level 3 - Model-Level
```
../KernelBench/KernelBench/level3/
```
- Complete models: GPT variants, ViT, etc.
- Full hierarchical decomposition: Model→Layer→Fusion→Kernel
- Most comprehensive test of decomposition

## Recommended Models for Testing

### Simple (Start Here)
1. `level2/conv_bn_relu.py` - Conv+BN+ReLU fusion
2. `level2/linear_gelu.py` - Linear+GELU fusion

### Medium Complexity
1. `level3/simple_transformer.py` - Basic transformer
2. `level3/vit_block.py` - Vision transformer block

### Complex (Full Test)
1. `level3/gpt_oss.py` - GPT-style model with MoE
2. `level3/llama.py` - LLaMA-style model

## Usage with Agent

```
I need you to decompose a PyTorch model from KernelBench.

## Instructions
Follow the methodology in: prompts/MAIN_PROMPT.md

## Model
The model is located at: ../KernelBench/KernelBench/level3/gpt_oss.py

## Requirements
1. Read the model code
2. Analyze the architecture
3. Decompose hierarchically: Model → Layer → Fusion → Kernel
4. Create all component files in output/gpt_oss/
5. Create verification tests
6. Verify all tests pass

## Output Location
output/gpt_oss/
```

## Notes on gpt_oss.py

The `gpt_oss.py` model has external dependencies. When decomposing:

1. **Extract the core Model class** - Focus on the computation graph
2. **Ignore config loading** - Use hardcoded values for testing
3. **Create self-contained components** - Each decomposed file should be standalone

Expected decomposition for gpt_oss.py:
```
gpt_oss (Level 3 - Model)
├── Embedding (Level 0 - Kernel)
├── TransformerLayer x N (Level 2 - Layer)
│   ├── AttentionBlock (Level 1 - Fusion)
│   │   ├── RMSNorm (Level 0 - Kernel)
│   │   ├── QKV Linear (Level 0 - Kernel)
│   │   ├── RoPE (Level 0 - Kernel)
│   │   ├── SDPA (Level 0 - Kernel)
│   │   ├── Output Linear (Level 0 - Kernel)
│   │   └── Residual Add (Level 0 - Kernel)
│   └── MoEBlock (Level 1 - Fusion)
│       ├── RMSNorm (Level 0 - Kernel)
│       ├── Router (TopK + Softmax) (Level 0 - Kernel)
│       ├── Expert Linear (Level 0 - Kernel)
│       ├── SwiGLU (Level 0 - Kernel)
│       ├── Expert Linear (Level 0 - Kernel)
│       └── Residual Add (Level 0 - Kernel)
└── OutputHead (Level 1 - Fusion)
    ├── RMSNorm (Level 0 - Kernel)
    └── Linear (Level 0 - Kernel)
```
