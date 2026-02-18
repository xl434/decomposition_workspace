#!/usr/bin/env python3
"""
Run All Component Tests

Runs all component tests in a decomposition output directory
and reports pass/fail status.

Usage:
    python run_all_tests.py <decomposition_dir>

Example:
    python run_all_tests.py output/gpt2/
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def find_all_components(decomp_dir: Path) -> List[Path]:
    """Find all Python component files in the decomposition."""
    components = []

    for level_dir in ["level_0_kernel", "level_1_fusion", "level_2_layer", "level_3_model"]:
        level_path = decomp_dir / level_dir
        if level_path.exists():
            for py_file in level_path.glob("*.py"):
                if not py_file.name.startswith("__"):
                    components.append(py_file)

    return sorted(components)


def run_component_test(component_path: Path, timeout: int = 60) -> Tuple[bool, str]:
    """
    Run a component's test.

    Returns:
        (passed, output_message)
    """
    try:
        result = subprocess.run(
            [sys.executable, str(component_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=component_path.parent,
        )

        output = result.stdout + result.stderr

        if result.returncode == 0 and "PASS" in output:
            return True, "PASS"
        else:
            # Extract error message
            error_lines = [l for l in output.split("\n") if "FAIL" in l or "Error" in l]
            error_msg = error_lines[0] if error_lines else f"Exit code {result.returncode}"
            return False, error_msg[:80]

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:80]


def run_all_tests(decomp_dir: Path, timeout: int = 60) -> bool:
    """
    Run all component tests and report results.

    Args:
        decomp_dir: Decomposition output directory
        timeout: Timeout per test in seconds

    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 60)
    print("COMPONENT TEST RUNNER")
    print("=" * 60)
    print(f"Directory: {decomp_dir}")
    print()

    components = find_all_components(decomp_dir)
    print(f"Found {len(components)} component(s)")
    print()

    if not components:
        print("[WARNING] No components found!")
        return False

    passed = 0
    failed = 0
    results = []

    for i, component in enumerate(components, 1):
        # Get relative path for display
        rel_path = component.relative_to(decomp_dir)
        print(f"[{i}/{len(components)}] Testing {rel_path}...", end=" ", flush=True)

        success, message = run_component_test(component, timeout)

        if success:
            print("PASS")
            passed += 1
        else:
            print(f"FAIL - {message}")
            failed += 1

        results.append((rel_path, success, message))

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:  {len(components)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()

    if failed > 0:
        print("Failed components:")
        for path, success, message in results:
            if not success:
                print(f"  - {path}: {message}")
        print()

    if failed == 0:
        print("[PASS] All component tests passed!")
        return True
    else:
        print(f"[FAIL] {failed} component(s) failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all component tests")
    parser.add_argument(
        "decomp_dir",
        type=Path,
        help="Path to decomposition output directory"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per test in seconds (default: 60)"
    )

    args = parser.parse_args()

    if not args.decomp_dir.exists():
        print(f"Error: Directory not found: {args.decomp_dir}")
        sys.exit(2)

    success = run_all_tests(args.decomp_dir, args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
