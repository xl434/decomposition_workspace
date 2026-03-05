#!/usr/bin/env python3
"""
Coverage Analysis Script

Extracts all operations from a PyTorch model and compares against decomposed
components to verify coverage completeness.

Uses a three-tier fallback strategy:
1. torch.fx.symbolic_trace + ShapeProp (most detailed)
2. torch.compile with capture backend (handles control flow)
3. Forward hooks on leaf modules (always works, misses functional ops)

Usage:
    python scripts/extract_ops.py \\
        --model path/to/original.py \\
        --decomp-dir path/to/output/model_name/ \\
        [--output path/to/coverage_report.json] \\
        [--children path/to/child1.py path/to/child2.py ...]

Exit codes:
    0 = Full coverage (100%)
    1 = Partial coverage (some ops missing)
    2 = Error (could not analyze model)
"""

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# =========================================================================
# Model loading
# =========================================================================

def load_model_from_file(path: Path, module_name: str = "module"):
    """Load a model and its inputs from a Python file."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    spec.loader.exec_module(module)

    Model = getattr(module, "Model", None)
    if Model is None:
        # Try RefactoredModel
        Model = getattr(module, "RefactoredModel", None)
    if Model is None:
        raise AttributeError(f"No Model class found in {path}")

    get_inputs = getattr(module, "get_inputs")
    get_init_inputs = getattr(module, "get_init_inputs", lambda: [])

    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()

    return model, inputs


# =========================================================================
# Strategy 1: torch.fx symbolic tracing
# =========================================================================

def extract_ops_fx(model: nn.Module, inputs: list) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Use torch.fx.symbolic_trace to get the computation graph with shapes.

    Returns (ops_list, error_message). ops_list is None on failure.
    """
    try:
        import torch.fx
        from torch.fx.passes.shape_prop import ShapeProp
    except ImportError:
        return None, "torch.fx not available"

    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as e:
        return None, f"symbolic_trace failed: {e}"

    # Run shape propagation
    try:
        ShapeProp(traced).propagate(*inputs)
    except Exception as e:
        # Shape prop failed, but we still have the graph
        pass

    ops = []
    for node in traced.graph.nodes:
        if node.op in ("call_module", "call_function", "call_method"):
            op_info = {
                "op_type": _classify_node(node, model),
                "node_op": node.op,
                "name": node.name,
                "target": str(node.target),
            }

            # Try to get shape info from metadata
            if hasattr(node, "meta") and "tensor_meta" in node.meta:
                meta = node.meta["tensor_meta"]
                if hasattr(meta, "shape"):
                    op_info["output_shape"] = list(meta.shape)
                    op_info["dtype"] = str(meta.dtype)

            ops.append(op_info)

    return ops, None


def _classify_node(node, model):
    """Classify a torch.fx node to a canonical operation name."""
    if node.op == "call_module":
        try:
            parts = node.target.split(".")
            mod = model
            for p in parts:
                mod = getattr(mod, p)
            return type(mod).__name__
        except Exception:
            return str(node.target)
    elif node.op == "call_function":
        fn = node.target
        if hasattr(fn, "__name__"):
            return fn.__name__
        return str(fn)
    elif node.op == "call_method":
        return str(node.target)
    return "unknown"


# =========================================================================
# Strategy 2: torch.compile with capture backend
# =========================================================================

def extract_ops_compile(model: nn.Module, inputs: list) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Use torch.compile with a custom backend to capture the FX graph.
    Handles most control flow that plain symbolic_trace cannot.

    Returns (ops_list, error_message). ops_list is None on failure.
    """
    try:
        captured_graphs = []

        def capture_backend(gm, example_inputs):
            captured_graphs.append(gm)
            return gm.forward

        compiled = torch.compile(model, backend=capture_backend)
        with torch.no_grad():
            compiled(*inputs)

        if not captured_graphs:
            return None, "No graphs captured by torch.compile"

        ops = []
        for gm in captured_graphs:
            for node in gm.graph.nodes:
                if node.op in ("call_module", "call_function", "call_method"):
                    op_info = {
                        "op_type": _classify_compile_node(node, gm),
                        "node_op": node.op,
                        "name": node.name,
                        "target": str(node.target),
                    }

                    if hasattr(node, "meta") and "val" in node.meta:
                        val = node.meta["val"]
                        if isinstance(val, torch.Tensor):
                            op_info["output_shape"] = list(val.shape)
                            op_info["dtype"] = str(val.dtype)

                    ops.append(op_info)

        return ops, None

    except Exception as e:
        return None, f"torch.compile capture failed: {e}"


def _classify_compile_node(node, gm):
    """Classify a node from a compiled graph."""
    if node.op == "call_module":
        try:
            mod = gm.get_submodule(node.target)
            return type(mod).__name__
        except Exception:
            return str(node.target)
    elif node.op == "call_function":
        fn = node.target
        if hasattr(fn, "__name__"):
            return fn.__name__
        # Handle torch ops like torch.ops.aten.mm
        name = str(fn)
        if "aten." in name:
            return name.split("aten.")[-1].split(".")[0]
        return name
    elif node.op == "call_method":
        return str(node.target)
    return "unknown"


# =========================================================================
# Strategy 3: Forward hooks (fallback)
# =========================================================================

def extract_ops_hooks(model: nn.Module, inputs: list) -> Tuple[List[Dict], Optional[str]]:
    """
    Register forward hooks on all leaf modules and record actual tensor shapes.
    Always works but only captures nn.Module calls, not functional ops.

    Returns (ops_list, warning_message).
    """
    ops = []
    hooks = []

    def make_hook(name, module):
        def hook(mod, inp, out):
            in_shapes = []
            for x in inp:
                if isinstance(x, torch.Tensor):
                    in_shapes.append(list(x.shape))

            out_shapes = []
            if isinstance(out, torch.Tensor):
                out_shapes = [list(out.shape)]
                out_dtype = str(out.dtype)
            elif isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor):
                        out_shapes.append(list(o.shape))
                out_dtype = str(out[0].dtype) if out and isinstance(out[0], torch.Tensor) else "unknown"
            else:
                out_dtype = "unknown"

            in_dtype = "unknown"
            for x in inp:
                if isinstance(x, torch.Tensor):
                    in_dtype = str(x.dtype)
                    break

            ops.append({
                "op_type": type(mod).__name__,
                "node_op": "call_module",
                "name": name,
                "target": name,
                "input_shapes": in_shapes,
                "output_shape": out_shapes[0] if len(out_shapes) == 1 else out_shapes,
                "dtype": in_dtype,
            })

        return hook

    # Hook all leaf modules (modules with no children)
    for name, mod in model.named_modules():
        if name == "":
            continue
        children = list(mod.children())
        if len(children) == 0:
            hooks.append(mod.register_forward_hook(make_hook(name, mod)))

    with torch.no_grad():
        model(*inputs)

    for h in hooks:
        h.remove()

    warning = (
        "Hook-based extraction only captures nn.Module calls. "
        "Functional ops (F.relu, torch.matmul, etc.) are NOT captured. "
        "Coverage may be incomplete."
    )

    return ops, warning


# =========================================================================
# Combined extraction with fallback
# =========================================================================

def extract_ops(model: nn.Module, inputs: list) -> Dict[str, Any]:
    """
    Extract operations using the three-tier fallback strategy.

    Returns a dict with: method_used, ops, warnings, errors.
    """
    result = {
        "method_used": None,
        "ops": [],
        "warnings": [],
        "errors": [],
    }

    # Tier 1: torch.fx
    ops, error = extract_ops_fx(model, inputs)
    if ops is not None:
        result["method_used"] = "torch.fx.symbolic_trace"
        result["ops"] = ops
        return result
    result["errors"].append(f"torch.fx: {error}")

    # Tier 2: torch.compile
    ops, error = extract_ops_compile(model, inputs)
    if ops is not None:
        result["method_used"] = "torch.compile"
        result["ops"] = ops
        return result
    result["errors"].append(f"torch.compile: {error}")

    # Tier 3: forward hooks
    ops, warning = extract_ops_hooks(model, inputs)
    result["method_used"] = "forward_hooks"
    result["ops"] = ops
    if warning:
        result["warnings"].append(warning)

    return result


# =========================================================================
# Coverage comparison
# =========================================================================

# Map low-level ATen/functional op names to canonical kernel-level names.
# This bridges the gap between torch.compile (which captures ATen ops like
# "linear", "embedding") and forward hooks (which capture nn.Module names
# like "Linear", "Embedding").
OP_CANONICAL_MAP = {
    # nn.Module names → canonical
    "Linear": "linear",
    "Conv2d": "conv2d",
    "Conv1d": "conv1d",
    "Conv3d": "conv3d",
    "ConvTranspose2d": "conv_transpose2d",
    "BatchNorm2d": "batch_norm",
    "LayerNorm": "layer_norm",
    "GroupNorm": "group_norm",
    "Embedding": "embedding",
    "LSTM": "lstm",
    "GRU": "gru",
    "ReLU": "relu",
    "GELU": "gelu",
    "SiLU": "silu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softmax": "softmax",
    "Dropout": "dropout",
    "MaxPool2d": "max_pool2d",
    "AvgPool2d": "avg_pool2d",
    "AdaptiveAvgPool2d": "adaptive_avg_pool2d",
    # Already-lowered names (identity mapping)
    "linear": "linear",
    "conv2d": "conv2d",
    "conv1d": "conv1d",
    "embedding": "embedding",
    "layer_norm": "layer_norm",
    "group_norm": "group_norm",
    "gelu": "gelu",
    "silu": "silu",
    "softmax": "softmax",
    "matmul": "matmul",
    "bmm": "matmul",
    "mm": "matmul",
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "dropout": "dropout",
    "batch_norm": "batch_norm",
    # ATen-lowered names that torch.compile may produce
    "_native_batch_norm_legit": "batch_norm",
    "native_layer_norm": "layer_norm",
    "native_group_norm": "group_norm",
    "addmm": "linear",
    "_softmax": "softmax",
}

# Operations that are data plumbing / shape manipulation, NOT compute ops.
# These should be excluded from coverage comparison since they aren't
# expected to have corresponding kernel files.
#
# Also includes PRIMITIVE ops that are internal sub-operations of higher-level
# kernels. For example, when torch.compile traces through RMSNorm, it captures
# individual pow/mean/rsqrt/mul ops — but these are covered by the rms_norm
# kernel. Similarly, RoPE's cos/sin ops and attention's add/mul for masking
# are sub-operations, not standalone kernels.
DATA_PLUMBING_OPS = {
    "output", "placeholder", "get_attr", "unknown",
    # Shape/data manipulation
    "reshape", "view", "permute", "transpose", "contiguous", "flatten",
    "unsqueeze", "squeeze", "expand", "expand_as", "repeat",
    "cat", "stack", "split", "chunk", "narrow",
    "clone", "to", "type", "float", "half", "int",
    "getitem", "setitem",
    # Index/range generation
    "arange", "linspace", "ones", "zeros", "full", "empty_like",
    "ones_like", "zeros_like", "tensor",
    # Scalar/boolean ops for control flow and masking
    "le", "lt", "ge", "gt", "eq", "ne", "and_", "or_", "not_",
    "min", "max", "where",
    # In-place arithmetic
    "iadd", "imul",
    # Primitive arithmetic — these are sub-operations inside higher-level kernels
    # (e.g., pow/mean/rsqrt inside RMSNorm, cos/sin inside RoPE, add/mul for
    # residuals and scaling). They are NOT standalone kernel-level operations.
    "add", "sub", "mul", "truediv", "pow", "rsqrt", "sqrt",
    "cos", "sin", "exp", "log",
    "mean", "sum", "cumsum",
    # Loss functions (typically in model-level forward as arithmetic)
    "mse_loss",
}


def normalize_op_type(op_type: str) -> str:
    """Normalize an op type to its canonical name."""
    return OP_CANONICAL_MAP.get(op_type, op_type)


def is_compute_op(op_type: str) -> bool:
    """Return True if this op is a compute operation (not data plumbing)."""
    canonical = normalize_op_type(op_type)
    return canonical not in DATA_PLUMBING_OPS


def normalize_op(op: Dict) -> Tuple[str, Optional[str]]:
    """Normalize an op to (op_type, dtype) for comparison."""
    op_type = normalize_op_type(op.get("op_type", "unknown"))
    dtype = op.get("dtype", "unknown")
    return (op_type, dtype)


def extract_ops_from_children(
    children_paths: List[Path],
    decomp_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Extract ops from all child component files.

    Tries to load each child, instantiate its model, and extract ops via hooks.
    """
    all_ops = []

    for path in children_paths:
        if not path.exists():
            continue

        try:
            model, inputs = load_model_from_file(path, f"child_{path.stem}")
            ops, _ = extract_ops_hooks(model, inputs)
            for op in ops:
                op["source_file"] = str(path.name)
            all_ops.extend(ops)
        except Exception as e:
            all_ops.append({
                "op_type": "LOAD_ERROR",
                "source_file": str(path.name),
                "error": str(e),
            })

    return all_ops


def find_children_in_decomp(decomp_dir: Path) -> List[Path]:
    """Find leaf-level (L0 kernel) component files in a decomposition directory.

    Only scans level_0_kernel/ to avoid counting the same operation multiple
    times across hierarchy levels (L0, L1, L2 all contain the same ops nested).
    """
    children = []
    level_dir = decomp_dir / "level_0_kernel"
    if level_dir.exists():
        for py_file in sorted(level_dir.glob("*.py")):
            if not py_file.name.startswith("__"):
                children.append(py_file)
    return children


def compute_coverage(
    original_ops: List[Dict],
    decomposed_ops: List[Dict],
) -> Dict[str, Any]:
    """
    Compare original model ops against decomposed component ops.

    Uses CANONICAL op_type counting. Normalizes names (e.g., "Linear" ↔ "linear")
    and filters out data plumbing ops (reshape, view, cat, indexing, etc.) that
    don't have corresponding kernel files.
    """
    # Count COMPUTE ops by canonical type in original
    orig_counts = defaultdict(int)
    orig_plumbing = defaultdict(int)
    for op in original_ops:
        raw_type = op.get("op_type", "unknown")
        canonical = normalize_op_type(raw_type)
        if is_compute_op(raw_type):
            orig_counts[canonical] += 1
        else:
            orig_plumbing[canonical] += 1

    # Count COMPUTE ops by canonical type in decomposed
    decomp_counts = defaultdict(int)
    for op in decomposed_ops:
        raw_type = op.get("op_type", "unknown")
        if raw_type == "LOAD_ERROR":
            continue
        canonical = normalize_op_type(raw_type)
        if is_compute_op(raw_type):
            decomp_counts[canonical] += 1

    # Compare
    all_op_types = set(orig_counts.keys()) | set(decomp_counts.keys())

    covered = []
    missing = []
    extra = []
    op_details = []

    for op_type in sorted(all_op_types):
        orig_count = orig_counts.get(op_type, 0)
        decomp_count = decomp_counts.get(op_type, 0)

        detail = {
            "op_type": op_type,
            "original_count": orig_count,
            "decomposed_count": decomp_count,
        }

        if orig_count > 0 and decomp_count >= orig_count:
            detail["status"] = "covered"
            covered.append(op_type)
        elif orig_count > 0 and decomp_count > 0:
            detail["status"] = "partial"
            missing.append(op_type)
        elif orig_count > 0 and decomp_count == 0:
            detail["status"] = "missing"
            missing.append(op_type)
        else:
            detail["status"] = "extra"
            extra.append(op_type)

        op_details.append(detail)

    total_orig = sum(orig_counts.values())
    total_covered = sum(
        min(orig_counts.get(t, 0), decomp_counts.get(t, 0))
        for t in orig_counts
    )

    coverage_pct = (total_covered / total_orig * 100) if total_orig > 0 else 100.0

    return {
        "total_original_ops": total_orig,
        "total_decomposed_ops": sum(decomp_counts.values()),
        "covered_op_count": total_covered,
        "missing_op_types": missing,
        "extra_op_types": extra,
        "coverage_pct": round(coverage_pct, 1),
        "op_details": op_details,
        "data_plumbing_excluded": dict(orig_plumbing),
    }


# =========================================================================
# Main
# =========================================================================

def analyze_coverage(
    model_path: Path,
    decomp_dir: Optional[Path] = None,
    children_paths: Optional[List[Path]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full coverage analysis: extract ops from original, compare with decomposed.
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "model_file": str(model_path),
        "status": "UNKNOWN",
    }

    # Load and analyze original model
    print("=" * 60)
    print("COVERAGE ANALYSIS")
    print("=" * 60)
    print(f"Model: {model_path}")
    print()

    print("[1/3] Extracting ops from original model...")
    try:
        model, inputs = load_model_from_file(model_path, "original_coverage")
    except Exception as e:
        print(f"      [FAIL] Could not load model: {e}")
        result["status"] = "ERROR"
        result["error"] = str(e)
        _write_report(result, output_path)
        return result

    extraction = extract_ops(model, inputs)
    result["extraction_method"] = extraction["method_used"]
    result["extraction_warnings"] = extraction["warnings"]
    result["extraction_errors"] = extraction["errors"]
    result["original_ops"] = extraction["ops"]

    print(f"      Method: {extraction['method_used']}")
    print(f"      Found {len(extraction['ops'])} operations")
    if extraction["warnings"]:
        for w in extraction["warnings"]:
            print(f"      [WARN] {w}")

    # Summarize original ops (separated into compute vs plumbing)
    compute_summary = defaultdict(int)
    plumbing_summary = defaultdict(int)
    for op in extraction["ops"]:
        raw_type = op.get("op_type", "unknown")
        canonical = normalize_op_type(raw_type)
        if is_compute_op(raw_type):
            compute_summary[canonical] += 1
        elif canonical not in ("output", "placeholder", "get_attr", "unknown"):
            plumbing_summary[canonical] += 1

    if compute_summary:
        print("      Compute ops (will be matched against kernels):")
        for op_type, count in sorted(compute_summary.items()):
            print(f"        {op_type}: {count}")
    if plumbing_summary:
        print(f"      Data plumbing ops (excluded from coverage): "
              f"{sum(plumbing_summary.values())} total "
              f"({len(plumbing_summary)} types)")

    # Find and analyze children
    print("\n[2/3] Extracting ops from decomposed components...")

    if children_paths:
        child_files = children_paths
    elif decomp_dir:
        child_files = find_children_in_decomp(decomp_dir)
    else:
        print("      [SKIP] No children or decomp directory specified")
        result["status"] = "PARTIAL"
        result["coverage"] = None
        _write_report(result, output_path)
        return result

    print(f"      Found {len(child_files)} component files")

    decomposed_ops = extract_ops_from_children(child_files, decomp_dir)
    result["decomposed_ops_count"] = len(decomposed_ops)
    print(f"      Extracted {len(decomposed_ops)} total operations")

    # Compare coverage
    print("\n[3/3] Computing coverage...")

    coverage = compute_coverage(extraction["ops"], decomposed_ops)
    result["coverage"] = coverage

    print(f"      Coverage: {coverage['coverage_pct']}%")
    print(f"      Original ops: {coverage['total_original_ops']}")
    print(f"      Decomposed ops: {coverage['total_decomposed_ops']}")
    print(f"      Covered: {coverage['covered_op_count']}")

    if coverage["missing_op_types"]:
        print(f"      Missing types: {coverage['missing_op_types']}")
    if coverage["extra_op_types"]:
        print(f"      Extra types: {coverage['extra_op_types']}")

    # Op-by-op table
    if coverage["op_details"]:
        print()
        print("      Op Coverage Table:")
        print(f"      {'Op Type':<25} {'Original':>10} {'Decomposed':>12} {'Status':>10}")
        print(f"      {'-'*25} {'-'*10} {'-'*12} {'-'*10}")
        for d in coverage["op_details"]:
            print(f"      {d['op_type']:<25} {d['original_count']:>10} "
                  f"{d['decomposed_count']:>12} {d['status']:>10}")

    # Final status
    if coverage["coverage_pct"] >= 100.0:
        result["status"] = "FULL_COVERAGE"
        print(f"\n[PASS] Full coverage ({coverage['coverage_pct']}%)")
    elif coverage["coverage_pct"] >= 80.0:
        result["status"] = "PARTIAL_COVERAGE"
        print(f"\n[WARN] Partial coverage ({coverage['coverage_pct']}%)")
    else:
        result["status"] = "LOW_COVERAGE"
        print(f"\n[FAIL] Low coverage ({coverage['coverage_pct']}%)")

    _write_report(result, output_path)
    return result


def _write_report(result: Dict, output_path: Optional[Path]):
    """Write report to JSON file."""
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a serializable version (remove raw op lists for readability)
        report = {k: v for k, v in result.items() if k != "original_ops"}

        # Summarize original ops using canonical names
        compute_ops = defaultdict(int)
        plumbing_ops = defaultdict(int)
        for op in result.get("original_ops", []):
            raw_type = op.get("op_type", "unknown")
            canonical = normalize_op_type(raw_type)
            if is_compute_op(raw_type):
                compute_ops[canonical] += 1
            elif canonical not in ("output", "placeholder", "get_attr", "unknown"):
                plumbing_ops[canonical] += 1

        report["original_compute_ops"] = dict(compute_ops)
        report["original_plumbing_ops"] = dict(plumbing_ops)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze operation coverage of a model decomposition"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to original model file",
    )
    parser.add_argument(
        "--decomp-dir",
        type=Path,
        default=None,
        help="Path to decomposition output directory (auto-finds children)",
    )
    parser.add_argument(
        "--children",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit paths to child component files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write coverage_report.json",
    )

    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(2)

    result = analyze_coverage(
        model_path=args.model,
        decomp_dir=args.decomp_dir,
        children_paths=args.children,
        output_path=args.output,
    )

    if result["status"] == "FULL_COVERAGE":
        sys.exit(0)
    elif result["status"] in ("PARTIAL_COVERAGE", "LOW_COVERAGE"):
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
