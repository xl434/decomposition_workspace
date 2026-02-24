#!/usr/bin/env python3
"""
Step-Level Verification Script

Verifies that a refactored module produces identical output to the original
module when sharing the same weights. This is the canonical verification tool
that agents call directly after each decomposition step.

Usage:
    python scripts/verify_step.py \\
        --original path/to/parent.py \\
        --refactored path/to/refactored.py \\
        [--weight-map path/to/weight_map.json] \\
        [--output path/to/verification_result.json] \\
        [--rtol 1e-5] [--atol 1e-6] \\
        [--num-trials 3] \\
        [--skip-anticheat]

The script performs:
1. Anti-cheat validation: ensures refactored code only calls child modules
2. Weight transfer: maps and copies weights from original to refactored
3. Numerical comparison: runs both models on same inputs, compares outputs

Exit codes:
    0 = PASS (all checks passed)
    1 = FAIL (numerical mismatch or anti-cheat violation)
    2 = ERROR (could not load models or other runtime error)
"""

import argparse
import ast
import importlib.util
import inspect
import json
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# =========================================================================
# Anti-cheat: disallowed patterns in refactored forward()
# =========================================================================

# Operations that MUST live in child modules, not in refactored forward()
DISALLOWED_CALL_PATTERNS = [
    # nn module constructors (should not appear in forward)
    r"nn\.Linear",
    r"nn\.Conv[123]d",
    r"nn\.ConvTranspose[123]d",
    r"nn\.BatchNorm[123]d",
    r"nn\.LayerNorm",
    r"nn\.GroupNorm",
    r"nn\.InstanceNorm[123]d",
    r"nn\.Embedding",
    r"nn\.LSTM",
    r"nn\.GRU",
    r"nn\.RNN",
    r"nn\.MultiheadAttention",
    r"nn\.Transformer",
    # Functional compute ops
    r"F\.linear",
    r"F\.conv[123]d",
    r"F\.conv_transpose[123]d",
    r"F\.batch_norm",
    r"F\.layer_norm",
    r"F\.group_norm",
    r"F\.instance_norm",
    r"F\.embedding",
    r"F\.relu",
    r"F\.gelu",
    r"F\.silu",
    r"F\.sigmoid",
    r"F\.tanh",
    r"F\.softmax",
    r"F\.log_softmax",
    r"F\.dropout",
    r"F\.scaled_dot_product_attention",
    r"F\.multi_head_attention_forward",
    r"F\.cross_entropy",
    r"F\.mse_loss",
    r"F\.binary_cross_entropy",
    # Torch compute ops
    r"torch\.matmul",
    r"torch\.bmm",
    r"torch\.mm",
    r"torch\.addmm",
    r"torch\.einsum",
    r"torch\.softmax",
    r"torch\.sigmoid",
    r"torch\.tanh",
    r"torch\.exp",
    r"torch\.log",
    r"torch\.sqrt",
    r"torch\.pow",
    r"torch\.clamp",
    r"torch\.where",
]

# Operations allowed in refactored forward() (data plumbing / shape ops)
# These don't need to be checked - anything NOT in DISALLOWED is allowed.
# Listed here for documentation:
# - self.child_module(x)       # child calls
# - x + y, x - y, x * y, x / y  # elementwise arithmetic (for residuals, scaling)
# - torch.cat, torch.stack, torch.split, torch.chunk  # tensor assembly
# - x.reshape, x.view, x.permute, x.transpose  # shape manipulation
# - x.contiguous, x.flatten, x.unsqueeze, x.squeeze  # shape manipulation
# - x[:, :, 0], x[..., :n]    # indexing/slicing
# - x.to(dtype), x.float(), x.half()  # dtype casting


def check_anticheat_source(refactored_path: Path) -> List[Dict[str, Any]]:
    """
    Scan the refactored module's forward() source for disallowed operations.

    Returns a list of violations, each with pattern, line_number, and line_text.
    Empty list means no violations found.
    """
    violations = []

    try:
        source = refactored_path.read_text(encoding="utf-8")
    except Exception as e:
        return [{"pattern": "FILE_READ_ERROR", "line_number": 0, "line_text": str(e)}]

    # Parse AST to find the forward() method
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [{"pattern": "SYNTAX_ERROR", "line_number": e.lineno or 0, "line_text": str(e)}]

    # Find the RefactoredModel class and its forward method
    forward_source = None
    forward_start_line = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "Refactored" in node.name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                    forward_start_line = item.lineno
                    forward_end_line = item.end_lineno or (item.lineno + 100)
                    source_lines = source.split("\n")
                    forward_source = "\n".join(
                        source_lines[forward_start_line - 1 : forward_end_line]
                    )
                    break

    if forward_source is None:
        # Try looking for any class with a forward method
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                        forward_start_line = item.lineno
                        forward_end_line = item.end_lineno or (item.lineno + 100)
                        source_lines = source.split("\n")
                        forward_source = "\n".join(
                            source_lines[forward_start_line - 1 : forward_end_line]
                        )
                        break

    if forward_source is None:
        return [{"pattern": "NO_FORWARD_METHOD", "line_number": 0,
                 "line_text": "Could not find forward() method in refactored code"}]

    # Scan forward() source for disallowed patterns
    forward_lines = forward_source.split("\n")
    for i, line in enumerate(forward_lines):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        for pattern in DISALLOWED_CALL_PATTERNS:
            if re.search(pattern, line):
                violations.append({
                    "pattern": pattern,
                    "line_number": (forward_start_line or 1) + i,
                    "line_text": stripped,
                })

    return violations


def check_anticheat_parameters(refactored_model: nn.Module) -> List[Dict[str, str]]:
    """
    Check that all parameters in the refactored model belong to named child submodules.

    Returns list of standalone parameter violations (parameters not under any child).
    """
    violations = []

    # Get names of direct child modules
    child_names = set()
    for name, _ in refactored_model.named_children():
        child_names.add(name)

    # Check each parameter
    for param_name, _ in refactored_model.named_parameters():
        # A parameter like "child_a.linear.weight" is fine (belongs to child_a)
        # A parameter like "standalone_weight" is a violation
        top_level = param_name.split(".")[0]
        if top_level not in child_names:
            violations.append({
                "parameter": param_name,
                "reason": f"Parameter '{param_name}' does not belong to any child module. "
                          f"Known children: {sorted(child_names)}",
            })

    return violations


# =========================================================================
# Model loading
# =========================================================================

def load_model_from_file(
    path: Path,
    module_name: str = "module",
    model_class_name: str = None,
) -> Tuple[nn.Module, list, callable]:
    """
    Load a model from a Python file.

    Looks for:
    - model_class_name (if specified), else 'RefactoredModel', else 'Model'
    - get_inputs() function
    - get_init_inputs() function (optional, defaults to lambda: [])

    Returns (model_instance, inputs, get_inputs_fn)
    """
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    # Add parent directory to sys.path for relative imports
    parent_dir = str(path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    spec.loader.exec_module(module)

    # Find model class
    class_names_to_try = []
    if model_class_name:
        class_names_to_try.append(model_class_name)
    class_names_to_try.extend(["RefactoredModel", "Model"])

    ModelClass = None
    for name in class_names_to_try:
        ModelClass = getattr(module, name, None)
        if ModelClass is not None:
            break

    if ModelClass is None:
        raise AttributeError(
            f"No model class found in {path}. Tried: {class_names_to_try}"
        )

    get_inputs = getattr(module, "get_inputs", None)
    if get_inputs is None:
        raise AttributeError(f"No get_inputs() function found in {path}")

    get_init_inputs = getattr(module, "get_init_inputs", lambda: [])

    model = ModelClass(*get_init_inputs())
    inputs = get_inputs()

    return model, inputs, get_inputs


# =========================================================================
# Weight mapping and transfer
# =========================================================================

def build_weight_map(
    original_sd: Dict[str, torch.Tensor],
    refactored_sd: Dict[str, torch.Tensor],
    explicit_map: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Build mapping from original state_dict keys to refactored state_dict keys.

    Strategy:
    1. If explicit_map provided, use it directly
    2. Try exact name match
    3. Try shape+dtype matching (greedy)

    Returns (mapping, unmapped_original_keys, unmapped_refactored_keys)
    """
    if explicit_map:
        unmapped_orig = [k for k in original_sd if k not in explicit_map]
        unmapped_ref = [
            k for k in refactored_sd if k not in explicit_map.values()
        ]
        return explicit_map, unmapped_orig, unmapped_ref

    mapping = {}
    ref_remaining = dict(refactored_sd)

    # Pass 1: exact name match
    for orig_key in list(original_sd.keys()):
        if orig_key in ref_remaining:
            mapping[orig_key] = orig_key
            del ref_remaining[orig_key]

    # Pass 2: shape+dtype match for remaining
    orig_remaining = {k: v for k, v in original_sd.items() if k not in mapping}
    for orig_key, orig_param in orig_remaining.items():
        for ref_key, ref_param in list(ref_remaining.items()):
            if (orig_param.shape == ref_param.shape
                    and orig_param.dtype == ref_param.dtype):
                mapping[orig_key] = ref_key
                del ref_remaining[ref_key]
                break

    unmapped_orig = [k for k in original_sd if k not in mapping]
    unmapped_ref = list(ref_remaining.keys())

    return mapping, unmapped_orig, unmapped_ref


def transfer_weights(
    original: nn.Module,
    refactored: nn.Module,
    weight_map: Dict[str, str],
) -> None:
    """Copy weights from original to refactored using the mapping."""
    orig_sd = original.state_dict()
    ref_sd = refactored.state_dict()

    for orig_key, ref_key in weight_map.items():
        if orig_key in orig_sd and ref_key in ref_sd:
            ref_sd[ref_key] = orig_sd[orig_key].clone()

    refactored.load_state_dict(ref_sd)


# =========================================================================
# Numerical comparison
# =========================================================================

def detect_dtype(inputs: list) -> torch.dtype:
    """Detect the primary dtype from inputs."""
    for inp in inputs:
        if isinstance(inp, torch.Tensor) and inp.is_floating_point():
            return inp.dtype
    return torch.float32


def get_tolerance(dtype: torch.dtype, rtol: float = None, atol: float = None):
    """Get appropriate tolerance for the given dtype."""
    if rtol is not None and atol is not None:
        return rtol, atol

    if dtype in (torch.float16, torch.bfloat16):
        return (rtol or 1e-3, atol or 1e-3)
    else:  # float32, float64
        return (rtol or 1e-5, atol or 1e-6)


def compare_outputs(
    original_out: Any,
    refactored_out: Any,
    rtol: float,
    atol: float,
) -> Tuple[bool, float, List[Dict]]:
    """
    Compare model outputs.

    Returns (all_match, max_diff, per_output_details)
    """
    # Normalize to list of tensors
    def to_tensor_list(out):
        if isinstance(out, torch.Tensor):
            return [out]
        elif isinstance(out, (tuple, list)):
            tensors = []
            for o in out:
                if isinstance(o, torch.Tensor):
                    tensors.append(o)
            return tensors
        else:
            raise TypeError(f"Unsupported output type: {type(out)}")

    orig_tensors = to_tensor_list(original_out)
    ref_tensors = to_tensor_list(refactored_out)

    if len(orig_tensors) != len(ref_tensors):
        return False, float("inf"), [{
            "error": f"Output count mismatch: original={len(orig_tensors)}, "
                     f"refactored={len(ref_tensors)}"
        }]

    all_match = True
    max_diff = 0.0
    details = []

    for i, (o, r) in enumerate(zip(orig_tensors, ref_tensors)):
        if o.shape != r.shape:
            all_match = False
            details.append({
                "output_idx": i,
                "shape_match": False,
                "original_shape": list(o.shape),
                "refactored_shape": list(r.shape),
            })
            continue

        # Compare in float32 for consistent comparison
        o_f = o.float()
        r_f = r.float()
        diff = (o_f - r_f).abs().max().item()
        max_diff = max(max_diff, diff)
        matches = torch.allclose(o_f, r_f, rtol=rtol, atol=atol)

        if not matches:
            all_match = False

        details.append({
            "output_idx": i,
            "shape_match": True,
            "shape": list(o.shape),
            "dtype": str(o.dtype),
            "max_diff": diff,
            "mean_diff": (o_f - r_f).abs().mean().item(),
            "matches": matches,
        })

    return all_match, max_diff, details


# =========================================================================
# Main verification
# =========================================================================

def verify_step(
    original_path: Path,
    refactored_path: Path,
    weight_map_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    num_trials: int = 3,
    skip_anticheat: bool = False,
) -> Dict[str, Any]:
    """
    Run full step verification.

    Returns a result dict with status, details, and violations.
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "original_file": str(original_path),
        "refactored_file": str(refactored_path),
        "status": "UNKNOWN",
        "anticheat": {"status": "SKIPPED", "violations": []},
        "weight_transfer": {"status": "UNKNOWN"},
        "numerical_comparison": {"status": "UNKNOWN"},
    }

    # ---- Step 1: Anti-cheat checks ----
    if not skip_anticheat:
        print("=" * 60)
        print("STEP VERIFICATION")
        print("=" * 60)
        print(f"Original:   {original_path}")
        print(f"Refactored: {refactored_path}")
        print()

        print("[1/3] Anti-cheat validation...")

        # Source scan
        source_violations = check_anticheat_source(refactored_path)
        if source_violations:
            print(f"      [WARN] {len(source_violations)} source violation(s) found:")
            for v in source_violations[:5]:
                print(f"        Line {v['line_number']}: {v['line_text']}")
                print(f"          Matched: {v['pattern']}")
            if len(source_violations) > 5:
                print(f"        ... and {len(source_violations) - 5} more")

        result["anticheat"]["source_violations"] = source_violations
    else:
        print("=" * 60)
        print("STEP VERIFICATION (anti-cheat skipped)")
        print("=" * 60)
        print(f"Original:   {original_path}")
        print(f"Refactored: {refactored_path}")
        print()

    # ---- Step 2: Load models ----
    print("[2/3] Loading models and transferring weights...")

    try:
        original, orig_inputs, get_inputs_fn = load_model_from_file(
            original_path, "original_module", "Model"
        )
        original.eval()
        print(f"      Original model loaded. Params: {sum(p.numel() for p in original.parameters())}")
    except Exception as e:
        print(f"      [FAIL] Could not load original model: {e}")
        result["status"] = "ERROR"
        result["error"] = f"Failed to load original: {e}"
        _write_result(result, output_path)
        return result

    try:
        refactored, _, _ = load_model_from_file(
            refactored_path, "refactored_module"
        )
        refactored.eval()
        print(f"      Refactored model loaded. Params: {sum(p.numel() for p in refactored.parameters())}")
    except Exception as e:
        print(f"      [FAIL] Could not load refactored model: {e}")
        result["status"] = "ERROR"
        result["error"] = f"Failed to load refactored: {e}"
        _write_result(result, output_path)
        return result

    # Anti-cheat: parameter check (after loading)
    if not skip_anticheat:
        param_violations = check_anticheat_parameters(refactored)
        if param_violations:
            print(f"      [WARN] {len(param_violations)} parameter violation(s):")
            for v in param_violations[:3]:
                print(f"        {v['parameter']}: {v['reason']}")

        result["anticheat"]["parameter_violations"] = param_violations
        all_violations = source_violations + param_violations
        result["anticheat"]["status"] = "PASS" if not all_violations else "FAIL"
        result["anticheat"]["violations"] = all_violations

        if all_violations:
            print(f"      [FAIL] Anti-cheat: {len(all_violations)} violation(s)")
            result["status"] = "FAIL"
            result["fail_reason"] = "anticheat_violation"
            _write_result(result, output_path)
            return result
        else:
            print("      [OK] Anti-cheat passed")

    # Weight mapping
    explicit_map = None
    if weight_map_path and weight_map_path.exists():
        with open(weight_map_path, encoding="utf-8") as f:
            explicit_map = json.load(f)
        print(f"      Using explicit weight map ({len(explicit_map)} entries)")

    orig_sd = original.state_dict()
    ref_sd = refactored.state_dict()

    weight_map, unmapped_orig, unmapped_ref = build_weight_map(
        orig_sd, ref_sd, explicit_map
    )

    print(f"      Weight map: {len(weight_map)} mapped, "
          f"{len(unmapped_orig)} unmapped original, "
          f"{len(unmapped_ref)} unmapped refactored")

    if unmapped_orig:
        print(f"      [WARN] Unmapped original keys: {unmapped_orig[:5]}")
    if unmapped_ref:
        print(f"      [WARN] Unmapped refactored keys: {unmapped_ref[:5]}")

    result["weight_transfer"] = {
        "status": "OK",
        "mapped_count": len(weight_map),
        "unmapped_original": unmapped_orig,
        "unmapped_refactored": unmapped_ref,
        "weight_map": weight_map,
    }

    # Transfer weights
    try:
        transfer_weights(original, refactored, weight_map)
        print("      [OK] Weights transferred")
    except Exception as e:
        print(f"      [FAIL] Weight transfer failed: {e}")
        result["status"] = "ERROR"
        result["error"] = f"Weight transfer failed: {e}"
        _write_result(result, output_path)
        return result

    # ---- Step 3: Numerical comparison ----
    print("[3/3] Running numerical comparison...")

    input_dtype = detect_dtype(orig_inputs)
    effective_rtol, effective_atol = get_tolerance(input_dtype, rtol, atol)
    print(f"      Input dtype: {input_dtype}, rtol={effective_rtol}, atol={effective_atol}")
    print(f"      Running {num_trials} trial(s)...")

    trials = []
    overall_pass = True
    overall_max_diff = 0.0

    for trial_idx in range(num_trials):
        seed = 42 + trial_idx
        torch.manual_seed(seed)

        # Generate fresh inputs with same shapes
        trial_inputs = get_inputs_fn()

        with torch.no_grad():
            try:
                orig_out = original(*trial_inputs)
            except Exception as e:
                print(f"      [FAIL] Original model error on trial {trial_idx}: {e}")
                trials.append({"trial": trial_idx, "seed": seed, "error": str(e), "pass": False})
                overall_pass = False
                continue

            try:
                ref_out = refactored(*trial_inputs)
            except Exception as e:
                print(f"      [FAIL] Refactored model error on trial {trial_idx}: {e}")
                trials.append({"trial": trial_idx, "seed": seed, "error": str(e), "pass": False})
                overall_pass = False
                continue

            matches, max_diff, details = compare_outputs(
                orig_out, ref_out, effective_rtol, effective_atol
            )

            overall_max_diff = max(overall_max_diff, max_diff)
            if not matches:
                overall_pass = False

            trials.append({
                "trial": trial_idx,
                "seed": seed,
                "max_diff": max_diff,
                "pass": matches,
                "output_details": details,
            })

            status_str = "PASS" if matches else "FAIL"
            print(f"      Trial {trial_idx} (seed={seed}): {status_str} "
                  f"(max_diff={max_diff:.2e})")

    result["numerical_comparison"] = {
        "status": "PASS" if overall_pass else "FAIL",
        "rtol": effective_rtol,
        "atol": effective_atol,
        "input_dtype": str(input_dtype),
        "num_trials": num_trials,
        "max_diff_across_trials": overall_max_diff,
        "trials": trials,
    }

    # ---- Final status ----
    result["status"] = "PASS" if overall_pass else "FAIL"

    print()
    print("-" * 60)
    if overall_pass:
        print(f"[PASS] Step verification PASSED (max_diff={overall_max_diff:.2e})")
    else:
        print(f"[FAIL] Step verification FAILED (max_diff={overall_max_diff:.2e})")
    print("-" * 60)

    _write_result(result, output_path)
    return result


def _write_result(result: Dict, output_path: Optional[Path]):
    """Write result to JSON file if output_path is specified."""
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults written to: {output_path}")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify a decomposition step: original vs refactored model"
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Path to original (parent) model file",
    )
    parser.add_argument(
        "--refactored",
        type=Path,
        required=True,
        help="Path to refactored model file (must use child modules only)",
    )
    parser.add_argument(
        "--weight-map",
        type=Path,
        default=None,
        help="Path to explicit weight_map.json (optional, auto-detected if omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write verification_result.json",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance (auto-detected from dtype if omitted)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance (auto-detected from dtype if omitted)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="Number of random input trials (default: 3)",
    )
    parser.add_argument(
        "--skip-anticheat",
        action="store_true",
        help="Skip anti-cheat source/parameter validation",
    )

    args = parser.parse_args()

    if not args.original.exists():
        print(f"Error: Original file not found: {args.original}")
        sys.exit(2)
    if not args.refactored.exists():
        print(f"Error: Refactored file not found: {args.refactored}")
        sys.exit(2)

    result = verify_step(
        original_path=args.original,
        refactored_path=args.refactored,
        weight_map_path=args.weight_map,
        output_path=args.output,
        rtol=args.rtol,
        atol=args.atol,
        num_trials=args.num_trials,
        skip_anticheat=args.skip_anticheat,
    )

    if result["status"] == "PASS":
        sys.exit(0)
    elif result["status"] == "FAIL":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
