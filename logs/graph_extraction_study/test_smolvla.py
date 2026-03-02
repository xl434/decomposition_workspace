#!/usr/bin/env python3
"""
Test graph extraction on the REAL SmolVLA from lerobot.

Run with: conda activate lerobot && python test_smolvla.py

This script loads the actual VLAFlowMatching class from lerobot and tests
torch.fx, torch.compile, torch.export, and forward hooks on it.
"""

import gc
import sys
import time as time_module
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

# Add lerobot source to path
LEROBOT_SRC = str(Path(__file__).resolve().parent.parent.parent / "data" / "lerobot" / "src")
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching


# =========================================================================
# Thin wrapper: converts positional tensor args to the list format
# expected by VLAFlowMatching.forward()
# =========================================================================
class SmolVLAWrapper(nn.Module):
    """Minimal wrapper — no reimplementation, just interface adaptation."""

    def __init__(self, vla_model):
        super().__init__()
        self.vla = vla_model

    def forward(self, image, img_mask, lang_tokens, lang_masks, state, actions, noise, time):
        return self.vla.forward(
            images=[image],
            img_masks=[img_mask],
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            actions=actions,
            noise=noise,
            time=time,
        )


# =========================================================================
# Load the real SmolVLA
# =========================================================================
def load_smolvla():
    """Load the actual VLAFlowMatching from lerobot source."""
    config = SmolVLAConfig(
        vlm_model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        load_vlm_weights=False,
        freeze_vision_encoder=True,
        train_expert_only=True,
        attention_mode="cross_attn",
        num_vlm_layers=16,
        expert_width_multiplier=0.75,
        chunk_size=50,
        max_state_dim=32,
        max_action_dim=32,
    )
    inner_model = VLAFlowMatching(config)
    inner_model.eval()
    model = SmolVLAWrapper(inner_model)
    model.eval()

    # Construct dummy inputs matching the real workload
    B = 1
    vlm_with_expert = inner_model.vlm_with_expert
    vocab_size = vlm_with_expert.config.text_config.vocab_size

    image = torch.randn(B, 3, 512, 512)
    img_mask = torch.ones(B, dtype=torch.bool)
    lang_tokens = torch.randint(0, vocab_size, (B, 16))
    lang_masks = torch.ones(B, 16, dtype=torch.bool)
    state = torch.randn(B, 32)
    actions = torch.randn(B, 50, 32)
    noise = torch.randn(B, 50, 32)
    time_tensor = torch.rand(B) * 0.999 + 0.001

    inputs = [image, img_mask, lang_tokens, lang_masks, state, actions, noise, time_tensor]
    n_params = sum(p.numel() for p in model.parameters())
    return model, inputs, n_params


# =========================================================================
# Test methods
# =========================================================================

def test_fx(model, inputs):
    """Test torch.fx.symbolic_trace."""
    print("\n--- torch.fx.symbolic_trace ---")
    try:
        traced = torch.fx.symbolic_trace(model)
        n = sum(1 for node in traced.graph.nodes
                if node.op in ("call_module", "call_function", "call_method"))
        print(f"  OK: {n} ops traced")
        return n
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return None


def test_compile(model, inputs):
    """Test torch.compile with custom backend to capture graphs."""
    print("\n--- torch.compile ---")
    captured = []

    def backend(gm, example_inputs):
        captured.append(gm)
        return gm.forward

    try:
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            compiled(*inputs)

        total_ops = 0
        op_types = defaultdict(int)
        for gm in captured:
            for node in gm.graph.nodes:
                if node.op in ("call_function", "call_module", "call_method"):
                    total_ops += 1
                    if node.op == "call_function":
                        name = str(node.target)
                        if "aten." in name:
                            name = name.split("aten.")[-1].split(".")[0]
                        else:
                            name = getattr(node.target, "__name__", name)
                    else:
                        name = str(node.target)
                    op_types[name] += 1

        n_breaks = len(captured) - 1
        print(f"  OK: {total_ops} ops across {len(captured)} graphs ({n_breaks} graph breaks)")
        print(f"  Graph sizes:")
        for i, gm in enumerate(captured):
            n = sum(1 for node in gm.graph.nodes
                    if node.op in ("call_function", "call_module", "call_method"))
            print(f"    Graph {i+1}: {n} ops")

        print(f"  Top 15 op types:")
        for op, count in sorted(op_types.items(), key=lambda x: -x[1])[:15]:
            print(f"    {op}: {count}")
        if len(op_types) > 15:
            print(f"    ... and {len(op_types)-15} more types")

        return total_ops, n_breaks
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return None, None
    finally:
        torch._dynamo.reset()


def test_export(model, inputs):
    """Test torch.export (both strict and non-strict)."""
    print("\n--- torch.export ---")
    for strict in [True, False]:
        label = "strict" if strict else "non-strict"
        print(f"  Trying {label}...")
        try:
            exported = torch.export.export(model, tuple(inputs), strict=strict)
            ops = defaultdict(int)
            for node in exported.graph.nodes:
                if node.op == "call_function":
                    name = str(node.target)
                    if "aten." in name:
                        name = name.split("aten.")[-1].split(".")[0]
                    ops[name] += 1
            total = sum(ops.values())
            print(f"  OK ({label}): {total} ops captured")
            print(f"  Top 10 op types:")
            for op, count in sorted(ops.items(), key=lambda x: -x[1])[:10]:
                print(f"    {op}: {count}")
            return total
        except Exception as e:
            err = str(e)
            print(f"  FAIL ({label}): {type(e).__name__}")
            # Show key diagnostic lines
            for line in err.split("\n"):
                line = line.strip()
                if line and any(kw in line.lower() for kw in
                    ["could not guard", "data-dependent", "unsupported",
                     "caused by", "hint:", "torch.arange", "branching",
                     "control flow", "item()", "scalar"]):
                    print(f"    {line[:120]}")
    return None


def test_hooks(model, inputs):
    """Test forward hooks on leaf modules."""
    print("\n--- Forward hooks ---")
    hooks = []
    hook_ops = defaultdict(int)

    def make_hook(name):
        def hook(module, inp, out):
            hook_ops[type(module).__name__] += 1
        return hook

    for name, mod in model.named_modules():
        if name and len(list(mod.children())) == 0:
            hooks.append(mod.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(*inputs)
        total = sum(hook_ops.values())
        print(f"  OK: {total} module-level ops")
        print(f"  Top 15 module types:")
        for op, count in sorted(hook_ops.items(), key=lambda x: -x[1])[:15]:
            print(f"    {op}: {count}")
        return total
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return None
    finally:
        for h in hooks:
            h.remove()


def test_dynamo_explain(model, inputs):
    """Use torch._dynamo.explain() for detailed graph break analysis."""
    print("\n--- torch._dynamo.explain() ---")
    try:
        explanation = torch._dynamo.explain(model)(*inputs)
        n_graphs = explanation.graph_count
        n_breaks = getattr(explanation, 'break_count',
                          getattr(explanation, 'graph_break_count', n_graphs - 1))
        break_reasons = getattr(explanation, 'break_reasons',
                               getattr(explanation, 'graph_break_reasons', []))

        print(f"  Graphs: {n_graphs}, Graph breaks: {n_breaks}")

        if break_reasons:
            seen = set()
            for i, reason in enumerate(break_reasons):
                reason_str = str(reason)
                short = reason_str[:200]
                if short in seen:
                    continue
                seen.add(short)
                print(f"\n  Break #{i+1}:")
                for line in reason_str.split("\n"):
                    line = line.strip()
                    if line:
                        print(f"    {line[:100]}")

    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
    finally:
        torch._dynamo.reset()


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 70)
    print("  SmolVLA Graph Extraction Test (REAL lerobot model)")
    print("=" * 70)

    print("\nLoading SmolVLA from lerobot VLAFlowMatching...")
    t0 = time_module.time()
    model, inputs, n_params = load_smolvla()
    print(f"  Loaded in {time_module.time()-t0:.1f}s, {n_params/1e6:.0f}M params")

    # Run a sanity forward pass first
    print("\nSanity check: forward pass...")
    with torch.no_grad():
        out = model(*inputs)
    print(f"  Output shape: {out.shape}, dtype: {out.dtype}")

    # Test all methods
    test_fx(model, inputs)
    gc.collect()

    compile_ops, n_breaks = test_compile(model, inputs)
    gc.collect()

    test_export(model, inputs)
    gc.collect()

    test_hooks(model, inputs)
    gc.collect()

    # Detailed graph break analysis
    test_dynamo_explain(model, inputs)
    gc.collect()

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
