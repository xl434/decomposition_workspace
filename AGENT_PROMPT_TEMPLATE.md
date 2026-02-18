# Agent Prompt Template

Use this template when asking an AI agent to decompose a model.

---

## Template

```
I need you to hierarchically decompose a PyTorch model into unit tests.

## Instructions

Read and follow the decomposition methodology in:
decomposition_workspace/prompts/MAIN_PROMPT.md

## Verification Guide

Reference the verification guidelines in:
decomposition_workspace/prompts/VERIFICATION_GUIDE.md

## Output Schema

Read and follow the output organization schema in:
decomposition_workspace/OUTPUT_SCHEMA.md

## Model to Decompose

under data/kernelbench/level3 directory, example 2, 4, 11, 14, 18, 28, 29, 38, 39, 41

## Output Directory

decomposition_workspace/output/{source_level}/{model_name}/

## Required Deliverables

1. **Architecture Analysis**
   - Module hierarchy diagram
   - Data flow with shapes at each step
   - Classification of abstraction level

2. **Component Files**
   Create files in the output directory:
   - level_3_model/{name}.py (if applicable)
   - level_2_layer/{name}.py (if applicable)
   - level_1_fusion/{name}.py (if applicable)
   - level_0_kernel/{name}.py (required - all leaf components)

3. **Decomposition Tree**
   Create: output/{source_level}/{model_name}/decomposition_tree.json

4. **Verification**
   Create and run: output/{source_level}/{model_name}/verification/composition_test.py
   - Must demonstrate that composed kernels match original output

5. **Test Results**
   Run all component tests and report results

## Success Criteria

- [ ] All component files execute without error (print "PASS")
- [ ] All shapes are documented and verified
- [ ] **All dimensions match the original model EXACTLY** — do NOT reduce or simplify batch size, channel counts, hidden dimensions, kernel sizes, sequence lengths, or any other parameter. Use the exact values from the original model's declarations, `get_inputs()`, and `get_init_inputs()`
- [ ] composition_test.py PASSES (max_diff < 1e-4)
- [ ] decomposition_tree.json is complete
- [ ] Every leaf node is at kernel level (single operation)
- [ ] No abstraction levels are skipped

Begin decomposition:
```

---

## Example: Simple Transformer

```
I need you to hierarchically decompose a PyTorch model into unit tests.

## Instructions

Read and follow the decomposition methodology in:
decomposition_workspace/prompts/MAIN_PROMPT.md

## Model to Decompose

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mlp_dim = hidden_dim * mlp_ratio

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, hidden_dim)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Attention
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        x = self.proj(x)
        x = x + residual

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x + residual

        return x

def get_inputs():
    return [torch.randn(2, 32, 768)]

def get_init_inputs():
    return []
```

## Output Directory

decomposition_workspace/output/level2/simple_transformer/

## Required Deliverables

1. Architecture analysis with shapes
2. Component files at each level
3. decomposition_tree.json
4. verification/composition_test.py that PASSES
5. Test results showing all components pass

Begin decomposition:
```

---

## Example: Using KernelBench Model

```
I need you to hierarchically decompose a PyTorch model from KernelBench.

## Instructions

Read and follow the decomposition methodology in:
decomposition_workspace/prompts/MAIN_PROMPT.md

## Model Location

KernelBench/KernelBench/level3/gpt_oss.py

Please read this file and extract the core Model class for decomposition.
Note: The model has external dependencies - create self-contained components.

## Output Directory

decomposition_workspace/output/level3/gpt_oss/

## Required Deliverables

1. Architecture analysis with shapes
2. Component files at each level
3. decomposition_tree.json
4. verification/composition_test.py that PASSES
5. Test results showing all components pass

Begin decomposition:
```

---

## Tips for Better Results

1. **Be specific about output location** - Tell the agent exactly where to write files

2. **Request verification explicitly** - Ask for composition_test.py that passes

3. **Provide full context** - Include the prompt file path so agent can read it

4. **Iterate on failures** - If verification fails, ask agent to debug and fix

5. **Start simple** - Test with mlp_block.py before trying gpt_oss.py
