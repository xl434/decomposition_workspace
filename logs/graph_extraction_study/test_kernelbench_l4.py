#!/usr/bin/env python3
"""
Test torch.fx, torch.compile, torch.export, and forward hooks on
KernelBench Level 4 HuggingFace models.

Level 4 models (7 unique architectures):
  - GPT-2:    standard causal LM
  - GPT-Neo:  local + global attention patterns
  - OPT:      Meta's causal LM with different positional embeddings
  - BigBird:  sparse attention (block sparse + random + global tokens)
  - BART:     encoder-decoder (seq2seq) architecture
  - ELECTRA:  discriminative pre-training (replaced token detection)
  - Reformer: LSH (locality-sensitive hashing) attention

Usage:
    python test_kernelbench_l4.py
    python test_kernelbench_l4.py --model gpt_neo
    python test_kernelbench_l4.py --verbose
    python test_kernelbench_l4.py --quick  # skip large models
"""

import argparse
import gc
import sys
import time as time_module
from collections import defaultdict

import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_modules(model):
    return sum(1 for _ in model.named_modules()) - 1


# =========================================================================
# Method 1: torch.fx.symbolic_trace
# =========================================================================
def try_fx(model, inputs, input_kwargs=None):
    import torch.fx
    result = {"status": "UNKNOWN", "ops": 0, "op_types": {}, "error": None}
    try:
        traced = torch.fx.symbolic_trace(model)
        ops = defaultdict(int)
        for node in traced.graph.nodes:
            if node.op in ("call_module", "call_function", "call_method"):
                if node.op == "call_module":
                    try:
                        mod = traced.get_submodule(node.target)
                        ops[type(mod).__name__] += 1
                    except Exception:
                        ops[str(node.target)] += 1
                elif node.op == "call_function":
                    ops[getattr(node.target, "__name__", str(node.target))] += 1
                else:
                    ops[str(node.target)] += 1
        result["status"] = "OK"
        result["ops"] = sum(ops.values())
        result["op_types"] = dict(ops)
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return result


# =========================================================================
# Method 2: torch.compile with capture backend
# =========================================================================
def try_compile(model, inputs, input_kwargs=None):
    result = {"status": "UNKNOWN", "ops": 0, "op_types": {}, "error": None, "graph_breaks": 0}
    captured = []

    def backend(gm, example_inputs):
        captured.append(gm)
        return gm.forward

    try:
        compiled = torch.compile(model, backend=backend)
        with torch.no_grad():
            if input_kwargs:
                compiled(*inputs, **input_kwargs)
            else:
                compiled(*inputs)

        result["graph_breaks"] = len(captured) - 1
        ops = defaultdict(int)
        for gm in captured:
            for node in gm.graph.nodes:
                if node.op in ("call_module", "call_function", "call_method"):
                    if node.op == "call_function":
                        name = getattr(node.target, "__name__", str(node.target))
                        if "aten." in str(node.target):
                            name = str(node.target).split("aten.")[-1].split(".")[0]
                        ops[name] += 1
                    elif node.op == "call_module":
                        try:
                            mod = gm.get_submodule(node.target)
                            ops[type(mod).__name__] += 1
                        except Exception:
                            ops[str(node.target)] += 1
                    else:
                        ops[str(node.target)] += 1
        result["status"] = "OK"
        result["ops"] = sum(ops.values())
        result["op_types"] = dict(ops)
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return result


# =========================================================================
# Method 3: torch.export
# =========================================================================
def try_export(model, inputs, input_kwargs=None):
    result = {"status": "UNKNOWN", "ops": 0, "op_types": {}, "error": None}
    try:
        if input_kwargs:
            exported = torch.export.export(model, tuple(inputs), kwargs=input_kwargs)
        else:
            exported = torch.export.export(model, tuple(inputs))

        ops = defaultdict(int)
        for node in exported.graph.nodes:
            if node.op in ("call_function",):
                name = getattr(node.target, "__name__", str(node.target))
                if "aten." in str(node.target):
                    name = str(node.target).split("aten.")[-1].split(".")[0]
                ops[name] += 1
        result["status"] = "OK"
        result["ops"] = sum(ops.values())
        result["op_types"] = dict(ops)
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return result


# =========================================================================
# Method 4: Forward hooks
# =========================================================================
def try_hooks(model, inputs, input_kwargs=None):
    result = {"status": "UNKNOWN", "ops": 0, "op_types": {}, "error": None}
    hooks = []
    ops = defaultdict(int)

    def make_hook(name, mod):
        def hook(module, inp, out):
            ops[type(module).__name__] += 1
        return hook

    for name, mod in model.named_modules():
        if name == "":
            continue
        if len(list(mod.children())) == 0:
            hooks.append(mod.register_forward_hook(make_hook(name, mod)))

    try:
        with torch.no_grad():
            if input_kwargs:
                model(*inputs, **input_kwargs)
            else:
                model(*inputs)
        result["status"] = "OK"
        result["ops"] = sum(ops.values())
        result["op_types"] = dict(ops)
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    for h in hooks:
        h.remove()
    return result


# =========================================================================
# KernelBench Level 4 model loaders
# Uses from_config (no weight download) with smaller variants where needed.
# =========================================================================

def load_gpt2():
    """GPT-2 — standard causal LM (KernelBench L4: #7, #16, #19)."""
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(config).eval()
    inputs = [torch.randint(0, config.vocab_size, (1, 64))]
    return model, inputs, None, f"GPT-2 ({count_params(model)/1e6:.0f}M params, {count_modules(model)} modules)"


def load_gpt_neo():
    """GPT-Neo — local + global attention (KernelBench L4: #1, #3, #18).
    Uses 125M variant instead of 2.7B for memory."""
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_config(config).eval()
    inputs = [torch.randint(0, config.vocab_size, (1, 64))]
    return model, inputs, None, f"GPT-Neo-125M ({count_params(model)/1e6:.0f}M params, {count_modules(model)} modules)"


def load_opt():
    """OPT — Meta's causal LM (KernelBench L4: #2, #4, #8).
    Uses 125M variant instead of 1.3B for memory."""
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("facebook/opt-125m")
    model = AutoModelForCausalLM.from_config(config).eval()
    inputs = [torch.randint(0, config.vocab_size, (1, 64))]
    return model, inputs, None, f"OPT-125M ({count_params(model)/1e6:.0f}M params, {count_modules(model)} modules)"


def load_bigbird():
    """BigBird — sparse attention with block sparse + random + global tokens
    (KernelBench L4: #5, #9, #10)."""
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained("google/bigbird-roberta-base")
    # BigBird needs attention_type set for from_config
    config.attention_type = "original_full"  # avoid block_sparse issues on CPU
    model = AutoModel.from_config(config).eval()
    inputs = [torch.randint(0, config.vocab_size, (1, 64))]
    kwargs = {"attention_mask": torch.ones(1, 64, dtype=torch.long)}
    return model, inputs, kwargs, f"BigBird-RoBERTa ({count_params(model)/1e6:.0f}M params, {count_modules(model)} modules)"


def load_bart():
    """BART — encoder-decoder seq2seq (KernelBench L4: #6, #17, #20).
    Uses bart-base instead of bart-large for memory."""
    from transformers import AutoModelForSeq2SeqLM, AutoConfig
    config = AutoConfig.from_pretrained("facebook/bart-base")
    model = AutoModelForSeq2SeqLM.from_config(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 64))
    decoder_input_ids = torch.randint(0, config.vocab_size, (1, 32))
    inputs = [input_ids]
    kwargs = {"decoder_input_ids": decoder_input_ids}
    return model, inputs, kwargs, f"BART-base ({count_params(model)/1e6:.0f}M params, {count_modules(model)} modules)"


def load_electra():
    """ELECTRA — discriminative pre-training, replaced token detection
    (KernelBench L4: #11, #12, #14)."""
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained("google/electra-small-discriminator")
    model = AutoModel.from_config(config).eval()
    inputs = [torch.randint(0, config.vocab_size, (1, 64))]
    kwargs = {"attention_mask": torch.ones(1, 64, dtype=torch.long)}
    return model, inputs, kwargs, f"ELECTRA-small ({count_params(model)/1e6:.0f}M params, {count_modules(model)} modules)"


def load_reformer():
    """Reformer — LSH attention for long sequences (KernelBench L4: #13, #15).
    Note: Reformer requires sequence length divisible by bucket size."""
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained("google/reformer-enwik8")
    # Reduce for testing: shorter axial position shape
    config.axial_pos_shape = [16, 16]
    config.axial_pos_embds_dim = [config.hidden_size // 2, config.hidden_size // 2]
    model = AutoModelForCausalLM.from_config(config).eval()
    # Reformer uses character-level vocab (256 for enwik8)
    seq_len = 256  # must match product of axial_pos_shape
    inputs = [torch.randint(0, config.vocab_size, (1, seq_len))]
    return model, inputs, None, f"Reformer-enwik8 ({count_params(model)/1e6:.0f}M params, {count_modules(model)} modules)"


# =========================================================================
# Model registry
# =========================================================================

# KernelBench Level 4 unique architectures
L4_MODELS = {
    "gpt2":     load_gpt2,
    "gpt_neo":  load_gpt_neo,
    "opt":      load_opt,
    "bigbird":  load_bigbird,
    "bart":     load_bart,
    "electra":  load_electra,
    "reformer": load_reformer,
}

ALL_MODELS = L4_MODELS

# Quick mode: skip large/slow models
QUICK_MODELS = {"gpt2", "electra", "opt"}


# =========================================================================
# Reporting
# =========================================================================

def print_result(method, result, verbose=False):
    if result["status"] == "FAIL":
        print(f"    {method:<18} FAIL")
        err = result["error"]
        while err:
            print(f"      {err[:90]}")
            err = err[90:]
    else:
        extras = []
        if result.get("graph_breaks", 0) > 0:
            extras.append(f"{result['graph_breaks']} graph breaks")
        extra_str = f"  ({', '.join(extras)})" if extras else ""
        print(f"    {method:<18} OK    {result['ops']:>5} ops{extra_str}")
        if verbose and result["op_types"]:
            for op, count in sorted(result["op_types"].items(), key=lambda x: -x[1])[:15]:
                print(f"      {op}: {count}")
            if len(result["op_types"]) > 15:
                print(f"      ... and {len(result['op_types'])-15} more op types")


def main():
    parser = argparse.ArgumentParser(
        description="Graph extraction comparison on KernelBench L4 models")
    parser.add_argument("--model", type=str, default=None,
                       help=f"Test only this model. Options: {', '.join(ALL_MODELS.keys())}")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed op breakdown")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: only test small/fast models")
    args = parser.parse_args()

    print("=" * 70)
    print("  KernelBench Level 4 Graph Extraction Test")
    print("  Methods: torch.fx | torch.compile | torch.export | forward hooks")
    print("=" * 70)

    if args.model:
        if args.model not in ALL_MODELS:
            print(f"  Unknown model: {args.model}")
            print(f"  Available: {', '.join(ALL_MODELS.keys())}")
            sys.exit(1)
        models_to_test = {args.model: ALL_MODELS[args.model]}
    elif args.quick:
        models_to_test = {k: ALL_MODELS[k] for k in QUICK_MODELS if k in ALL_MODELS}
    else:
        models_to_test = ALL_MODELS

    all_results = []

    for name, loader in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"  Loading {name}...")

        try:
            model, inputs, kwargs, desc = loader()
        except Exception as e:
            print(f"  SKIP: Could not load model: {type(e).__name__}: {e}")
            continue

        print(f"  {desc}")
        print()

        row = {"model": name}

        # torch.fx
        print("  Testing torch.fx.symbolic_trace...")
        t0 = time_module.time()
        fx_result = try_fx(model, inputs, kwargs)
        row["fx"] = fx_result
        print_result("torch.fx", fx_result, args.verbose)

        # torch.compile
        print("  Testing torch.compile...")
        compile_result = try_compile(model, inputs, kwargs)
        row["compile"] = compile_result
        print_result("torch.compile", compile_result, args.verbose)

        # torch.export
        print("  Testing torch.export...")
        export_result = try_export(model, inputs, kwargs)
        row["export"] = export_result
        print_result("torch.export", export_result, args.verbose)

        # forward hooks
        print("  Testing forward hooks...")
        hooks_result = try_hooks(model, inputs, kwargs)
        row["hooks"] = hooks_result
        print_result("forward_hooks", hooks_result, args.verbose)

        all_results.append(row)

        # Cleanup
        del model
        torch._dynamo.reset()
        gc.collect()

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<18} {'torch.fx':>12} {'torch.compile':>16} {'torch.export':>15} {'hooks':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*16} {'-'*15} {'-'*10}")
    for r in all_results:
        def fmt(res):
            if res["status"] == "FAIL":
                return "FAIL"
            s = f"{res['ops']} ops"
            if res.get("graph_breaks", 0) > 0:
                s += f" ({res['graph_breaks']}gb)"
            return s

        print(f"  {r['model']:<18} {fmt(r['fx']):>12} {fmt(r['compile']):>16} "
              f"{fmt(r['export']):>15} {fmt(r['hooks']):>10}")

    # Success rate
    total = len(all_results)
    if total > 0:
        fx_ok = sum(1 for r in all_results if r['fx']['status'] == 'OK')
        co_ok = sum(1 for r in all_results if r['compile']['status'] == 'OK')
        ex_ok = sum(1 for r in all_results if r['export']['status'] == 'OK')
        ho_ok = sum(1 for r in all_results if r['hooks']['status'] == 'OK')
        print(f"\n  Success rate: fx={fx_ok}/{total}  compile={co_ok}/{total}  "
              f"export={ex_ok}/{total}  hooks={ho_ok}/{total}")

    # Level 4 architecture notes
    print(f"\n  Architecture notes:")
    print(f"  - GPT-2, GPT-Neo, OPT: causal LMs (decoder-only)")
    print(f"  - BigBird: sparse attention (block + random + global)")
    print(f"  - BART: encoder-decoder (cross-attention between enc/dec)")
    print(f"  - ELECTRA: discriminator (bidirectional encoder)")
    print(f"  - Reformer: LSH attention (locality-sensitive hashing)")


if __name__ == "__main__":
    main()
