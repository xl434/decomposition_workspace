#!/usr/bin/env python3
"""
Batch Test Script

Tests all decomposed outputs in the workspace.
Reports which decompositions pass/fail verification.

Usage:
    python batch_test.py [--level LEVEL] [--verbose]

Examples:
    python batch_test.py                    # Test all outputs
    python batch_test.py --level level3     # Test only level3 outputs
    python batch_test.py --verbose          # Show detailed output
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def find_decomposition_outputs(output_dir: Path, level_filter: str = None) -> List[Path]:
    """Find all decomposition output directories."""
    outputs = []

    for level_dir in output_dir.iterdir():
        if not level_dir.is_dir():
            continue
        if level_filter and level_dir.name != level_filter:
            continue

        for model_dir in level_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "decomposition_tree.json").exists():
                outputs.append(model_dir)

    return sorted(outputs)


def test_component(component_path: Path, timeout: int = 60) -> Tuple[bool, str]:
    """Run a single component test."""
    try:
        result = subprocess.run(
            [sys.executable, str(component_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=component_path.parent,
        )

        if result.returncode == 0 and "PASS" in result.stdout:
            return True, "PASS"
        else:
            error = result.stderr[:100] if result.stderr else result.stdout[:100]
            return False, error.strip()

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:100]


def test_decomposition(model_dir: Path, verbose: bool = False) -> Dict:
    """Test a complete decomposition."""
    result = {
        "model": model_dir.name,
        "level": model_dir.parent.name,
        "path": str(model_dir),
        "components_tested": 0,
        "components_passed": 0,
        "components_failed": 0,
        "composition_test": None,
        "status": "UNKNOWN",
        "errors": [],
    }

    # Find all component files
    components = []
    for level_subdir in ["level_0_kernel", "level_1_fusion", "level_2_layer", "level_3_model"]:
        level_path = model_dir / level_subdir
        if level_path.exists():
            components.extend(level_path.glob("*.py"))

    if verbose:
        print(f"  Found {len(components)} components")

    # Test each component
    for component in components:
        if component.name.startswith("__"):
            continue

        passed, msg = test_component(component)
        result["components_tested"] += 1

        if passed:
            result["components_passed"] += 1
        else:
            result["components_failed"] += 1
            result["errors"].append(f"{component.name}: {msg}")

            if verbose:
                print(f"    FAIL: {component.name} - {msg}")

    # Test composition if it exists
    composition_test = model_dir / "verification" / "composition_test.py"
    if composition_test.exists():
        passed, msg = test_component(composition_test, timeout=120)
        result["composition_test"] = "PASS" if passed else f"FAIL: {msg}"
    else:
        result["composition_test"] = "NOT_FOUND"

    # Determine overall status
    if result["components_failed"] == 0 and result["composition_test"] == "PASS":
        result["status"] = "PASSED"
    elif result["components_failed"] == 0:
        result["status"] = "PARTIAL"  # Components pass but no composition test
    else:
        result["status"] = "FAILED"

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch test decomposition outputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "output",
        help="Output directory to test"
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Filter by level (e.g., level1, level2, level3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write results to JSON file"
    )

    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Output directory not found: {args.output_dir}")
        print("No decompositions to test.")
        sys.exit(0)

    # Find all decomposition outputs
    outputs = find_decomposition_outputs(args.output_dir, args.level)

    if not outputs:
        print("No decomposition outputs found.")
        sys.exit(0)

    print("=" * 60)
    print("BATCH DECOMPOSITION TEST")
    print("=" * 60)
    print(f"Testing {len(outputs)} decomposition(s)")
    print()

    results = []
    passed = 0
    failed = 0
    partial = 0

    for i, model_dir in enumerate(outputs, 1):
        print(f"[{i}/{len(outputs)}] Testing {model_dir.parent.name}/{model_dir.name}...", end=" ")

        result = test_decomposition(model_dir, args.verbose)
        results.append(result)

        if result["status"] == "PASSED":
            print("PASSED")
            passed += 1
        elif result["status"] == "PARTIAL":
            print(f"PARTIAL ({result['components_passed']}/{result['components_tested']} components)")
            partial += 1
        else:
            print(f"FAILED ({result['components_failed']} failures)")
            failed += 1

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total:   {len(outputs)}")
    print(f"Passed:  {passed}")
    print(f"Partial: {partial}")
    print(f"Failed:  {failed}")

    if failed > 0:
        print()
        print("Failed decompositions:")
        for r in results:
            if r["status"] == "FAILED":
                print(f"  - {r['level']}/{r['model']}: {len(r['errors'])} error(s)")
                for err in r["errors"][:3]:
                    print(f"      {err}")
                if len(r["errors"]) > 3:
                    print(f"      ... and {len(r['errors']) - 3} more")

    # Write JSON output
    if args.output_json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "total": len(outputs),
            "passed": passed,
            "partial": partial,
            "failed": failed,
            "results": results,
        }
        args.output_json.write_text(json.dumps(output, indent=2))
        print(f"\nResults written to: {args.output_json}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
