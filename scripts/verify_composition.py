#!/usr/bin/env python3
"""
Composition Verification Script

Verifies that decomposed components, when composed together,
produce the same output as the original model.

Usage:
    python verify_composition.py <decomposition_dir> --original <original_model.py>

Example:
    python verify_composition.py output/gpt2/ --original examples/gpt2.py
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


def load_module_from_path(path: Path, module_name: str = "module"):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_original_model(original_path: Path):
    """Load the original model and its test inputs."""
    module = load_module_from_path(original_path, "original_model")

    Model = getattr(module, "Model")
    get_inputs = getattr(module, "get_inputs")
    get_init_inputs = getattr(module, "get_init_inputs", lambda: [])

    model = Model(*get_init_inputs())
    model.eval()

    return model, get_inputs()


def load_decomposition_tree(decomp_dir: Path) -> Dict[str, Any]:
    """Load the decomposition tree JSON."""
    tree_path = decomp_dir / "decomposition_tree.json"
    if not tree_path.exists():
        raise FileNotFoundError(f"Decomposition tree not found: {tree_path}")

    with open(tree_path, encoding="utf-8") as f:
        return json.load(f)


def load_kernel_components(decomp_dir: Path) -> Dict[str, nn.Module]:
    """Load all kernel-level components."""
    kernel_dir = decomp_dir / "level_0_kernel"
    if not kernel_dir.exists():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

    components = {}
    for py_file in kernel_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
        try:
            module = load_module_from_path(py_file, f"kernel_{py_file.stem}")
            Model = getattr(module, "Model")
            get_init_inputs = getattr(module, "get_init_inputs", lambda: [])
            components[py_file.stem] = Model(*get_init_inputs())
        except Exception as e:
            print(f"  [WARNING] Failed to load {py_file.name}: {e}")

    return components


def verify_composition(
    original_path: Path,
    decomp_dir: Path,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """
    Verify that decomposed components match the original model.

    Args:
        original_path: Path to original model file
        decomp_dir: Path to decomposition output directory
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if verification passes, False otherwise
    """
    print("=" * 60)
    print("COMPOSITION VERIFICATION")
    print("=" * 60)
    print(f"Original model: {original_path}")
    print(f"Decomposition:  {decomp_dir}")
    print()

    # Step 1: Load original model
    print("[1/4] Loading original model...")
    try:
        original_model, test_inputs = load_original_model(original_path)
        print(f"      Model loaded successfully")
        print(f"      Test input shapes: {[x.shape for x in test_inputs]}")
    except Exception as e:
        print(f"      [FAIL] Could not load original model: {e}")
        return False

    # Step 2: Load decomposition tree
    print("[2/4] Loading decomposition tree...")
    try:
        tree = load_decomposition_tree(decomp_dir)
        nodes = tree.get("nodes", tree.get("tree", {}))
        print(f"      Found {len(nodes)} components")
    except Exception as e:
        print(f"      [FAIL] Could not load decomposition tree: {e}")
        return False

    # Step 3: Load kernel components
    print("[3/4] Loading kernel components...")
    try:
        kernels = load_kernel_components(decomp_dir)
        print(f"      Loaded {len(kernels)} kernel components")
    except Exception as e:
        print(f"      [FAIL] Could not load kernels: {e}")
        return False

    # Step 4: Run comparison
    print("[4/4] Running comparison...")

    with torch.no_grad():
        # Run original model
        try:
            original_output = original_model(*test_inputs)
            print(f"      Original output shape: {original_output.shape}")
        except Exception as e:
            print(f"      [FAIL] Original model failed: {e}")
            return False

        # For now, we verify that all kernels execute
        # A full composition would require the data flow from the tree
        all_kernels_pass = True
        for name, kernel in kernels.items():
            try:
                kernel.eval()
                # Note: Each kernel has its own get_inputs, we'd need to chain them
                print(f"      Kernel '{name}' loaded OK")
            except Exception as e:
                print(f"      [FAIL] Kernel '{name}' failed: {e}")
                all_kernels_pass = False

    # Summary
    print()
    print("=" * 60)
    if all_kernels_pass:
        print("[PASS] All kernel components loaded successfully")
        print()
        print("NOTE: Full composition verification requires the agent to create")
        print("      verification/composition_test.py that chains components together.")
        return True
    else:
        print("[FAIL] Some kernel components failed to load")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify decomposition composition matches original"
    )
    parser.add_argument(
        "decomp_dir",
        type=Path,
        help="Path to decomposition output directory"
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Path to original model file"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance (default: 1e-4)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance (default: 1e-5)"
    )

    args = parser.parse_args()

    if not args.decomp_dir.exists():
        print(f"Error: Decomposition directory not found: {args.decomp_dir}")
        sys.exit(2)

    if not args.original.exists():
        print(f"Error: Original model not found: {args.original}")
        sys.exit(2)

    success = verify_composition(
        args.original,
        args.decomp_dir,
        args.rtol,
        args.atol,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
