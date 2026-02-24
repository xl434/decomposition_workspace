# Decomposition Verification Guide

## Quick Reference: Verification Steps

### 1. Individual Component Verification
```bash
# Each component must pass its own test
python level_0_kernel/linear.py      # → PASS
python level_1_fusion/attention.py   # → PASS
python level_2_layer/transformer.py  # → PASS
```

### 2. Composition Verification (Most Important!)

The key insight: **If decomposition is correct, recomposing all components should produce identical output to the original model.**

```python
# Pseudocode for composition test
original_model = load_original()
composed_model = compose_from_decomposed_parts()

# Use SAME weights
composed_model.load_state_dict(original_model.state_dict())

input = generate_test_input()
original_output = original_model(input)
composed_output = composed_model(input)

# These MUST match within numerical tolerance
assert torch.allclose(original_output, composed_output, rtol=1e-4, atol=1e-5)
```

### 3. Shape Flow Verification

Track tensor shapes through the decomposition:

```
Input [2, 32] (tokens)
  │
  ▼ embedding
[2, 32, 768]
  │
  ▼ layer_norm
[2, 32, 768]
  │
  ▼ attention (qkv_proj → attention → out_proj)
[2, 32, 768]
  │
  ▼ add (residual)
[2, 32, 768]
  │
  ... continue for all components ...
  │
  ▼ output_head
[2, 32, 50000] (logits)
```

### 4. Operation Coverage Verification

Count operations in original vs **leaf-level (L0 kernel) components only**. Each op type count should be 1:1 between original and decomposed. If decomposed counts are 2x or 3x the original, you are incorrectly counting ops from multiple hierarchy levels (L0 + L1 + L2) — only L0 kernels should be counted.

Count operations in original vs decomposed:

| Operation | Original Count | Decomposed Count | Match? |
|-----------|---------------|------------------|--------|
| embedding | 1 | 1 | ✓ |
| linear | 6 | 6 | ✓ |
| layer_norm | 4 | 4 | ✓ |
| softmax | 2 | 2 | ✓ |
| add | 4 | 4 | ✓ |

---

## Common Verification Failures and Fixes

### Failure: "Composed output doesn't match original"

**Possible causes:**
1. **Missing operation** - Check operation coverage
2. **Wrong order** - Verify data flow matches original forward()
3. **Shape mismatch** - An intermediate reshape is wrong
4. **Weight mismatch** - Weights not properly transferred

**Debug approach:**
```python
# Add intermediate checkpoints
def composed_forward_with_debug(self, x):
    x = self.embedding(x)
    print(f"After embedding: {x.shape}, mean={x.mean():.4f}")

    x = self.layer_norm(x)
    print(f"After norm: {x.shape}, mean={x.mean():.4f}")

    # Compare with original at each step
```

### Failure: "Dimensions don't match original model"

**Cause:** Agent reduced dimensions (e.g., batch=2 instead of 128, hidden=64 instead of 768) for "faster testing". This is WRONG.

**Fix:** Go back to the original model's source code, read the exact values from `__init__()`, `get_inputs()`, and `get_init_inputs()`, and use those exact values in all component files.

### Failure: "Shape mismatch at component X"

**Possible causes:**
1. **Wrong input shape** - Check get_inputs() returns correct shape matching the ORIGINAL model
2. **Wrong weight dimensions** - Linear(in, out) dimensions swapped
3. **Missing reshape/transpose** - Attention often needs these

### Failure: "Dtype mismatch"

**Fix:**
```python
def forward(self, x):
    # Ensure weights match input dtype
    if x.dtype != self.linear.weight.dtype:
        self.linear = self.linear.to(x.dtype)
    return self.linear(x)
```

---

## Abstraction Level Quick Reference

### Level 3: Model
- **Contains**: Multiple layers, full architecture
- **Examples**: GPT-2, ViT, LLaMA, BERT
- **Decomposes to**: Layers (embedding, transformer blocks, output head)

### Level 2: Layer
- **Contains**: 5-15 operations, a logical unit
- **Examples**: TransformerBlock, ResNetBlock, AttentionLayer
- **Decomposes to**: Fusions (attention, MLP, normalization groups)

### Level 1: Fusion
- **Contains**: 2-5 tightly coupled operations
- **Examples**: Conv+BN+ReLU, QKV projection, SwiGLU, Attention scores
- **Decomposes to**: Kernels (individual operations)

### Level 0: Kernel
- **Contains**: Single operation
- **Examples**: Linear, Conv2d, ReLU, LayerNorm, MatMul, Softmax
- **Decomposes to**: Nothing (leaf node)

---

---

## Step-by-Step Verification Protocol

### Per-Step Verification (REQUIRED at every level transition)

At each decomposition step, you MUST:

1. **Create refactored.py** — parent rewritten using ONLY child module calls
2. **Run verify_step.py** — the standard verification script
3. **PASS before proceeding** — do not decompose the next level until current step passes

```bash
# Standard verification command
python scripts/verify_step.py \
    --original path/to/parent.py \
    --refactored steps/step_N_name/refactored.py \
    --output steps/step_N_name/verification_result.json
```

The script checks:
1. **Anti-cheat** — refactored forward() only calls child modules + data plumbing
2. **Weight transfer** — maps and copies weights from original to refactored
3. **Numerical comparison** — 3 trials with different random seeds

### Tolerance Guidelines by Dtype

| Source Dtype | rtol | atol | Notes |
|-------------|------|------|-------|
| float32 | 1e-5 | 1e-6 | Standard precision |
| float16 | 1e-3 | 1e-3 | Half precision accumulation |
| bfloat16 | 1e-3 | 1e-3 | Brain float precision |
| int64/int32 | 0 | 0 | Exact match for integer ops |

Tolerances are auto-detected by verify_step.py based on input dtype. Override with `--rtol` and `--atol` if needed.

### Anti-Cheat Rules

Refactored `forward()` may ONLY contain:

**Allowed (data plumbing):**
- `self.child_module(x)` — child module calls
- `x + residual`, `x * scale` — arithmetic for residuals/scaling
- `torch.cat(...)`, `torch.split(...)`, `torch.chunk(...)` — tensor assembly
- `x.reshape(...)`, `x.permute(...)`, `x.transpose(...)`, `x.view(...)`, `x.flatten(...)` — shape ops
- `x[:, :, 0]`, `x[..., :n]` — indexing/slicing
- `x.contiguous()`, `x.to(dtype)` — memory/type casting

**Disallowed (must be in child module):**
- `nn.Linear(...)`, `nn.Conv2d(...)` — any nn.Module construction
- `F.linear(...)`, `F.softmax(...)`, `F.relu(...)` — functional compute
- `torch.matmul(...)`, `torch.bmm(...)`, `torch.einsum(...)` — compute ops
- `F.layer_norm(...)`, `F.batch_norm(...)` — normalization
- Any parameterized operation not imported from a child file

### When Standard Script Doesn't Work

If your model has special components that verify_step.py can't handle:
1. Write a custom verification script
2. It MUST use the same tolerance logic (see table above)
3. It MUST perform weight transfer (same weights for both models)
4. It MUST output the same JSON format as verify_step.py
5. It MUST include anti-cheat checks

### Edge Case Handling

**Residual connections:**
Keep residual adds in the refactored forward(), not in children:
```python
def forward(self, x):
    residual = x
    x = self.attention_child(x)
    x = x + residual          # stays here, not in child
    return x
```

**Multiple outputs:**
Children that return tuples — unpack in refactored code:
```python
def forward(self, x):
    output, cache = self.child_with_cache(x)
    return output, cache
```

**Shared/tied weights:**
If two children share the same weight (e.g., tied embeddings), use `weight_map.json` to map the single original parameter to both locations.

**In-place operations:**
`ReLU(inplace=True)` and similar — the refactored version should match the original's behavior. For verification, both produce identical numerical results.

**Control flow in forward():**
Preserve conditionals in refactored code:
```python
def forward(self, x):
    x = self.main_path(x)
    if self.has_downsample:       # conditional preserved
        residual = self.downsample_child(residual)
    return x + residual
```

---

## Checklist Before Submission

```
[ ] Each decomposition step has refactored.py with ONLY child module calls
[ ] Each step's verify_step.py PASSES
[ ] All component files execute without error
[ ] composition_test.py PASSES
[ ] Shape flow is documented and verified
[ ] Operation counts match between original and decomposed
[ ] No kernel has more than 1 operation
[ ] All leaf nodes are at kernel level
[ ] Abstraction hierarchy is respected (L3→L2→L1→L0)
[ ] decomposition_tree.json is complete
[ ] All dtypes are preserved
[ ] ALL DIMENSIONS MATCH THE ORIGINAL MODEL EXACTLY
    - get_inputs() uses the same shapes as the original model
    - get_init_inputs() passes the same parameters as the original model
    - No dimensions were reduced or simplified (batch size, channels, hidden dims, etc.)
```
