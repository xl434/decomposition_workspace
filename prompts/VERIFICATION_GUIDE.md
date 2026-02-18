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

## Checklist Before Submission

```
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
