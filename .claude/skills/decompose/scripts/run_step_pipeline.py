#!/usr/bin/env python3
"""
Step Verification Pipeline

Post-hoc validation tool that reads a decomposition's tree, identifies all
parent->child edges, and runs verify_step.py + extract_ops.py for each
step that has a refactored.py.

Usage:
    python scripts/run_step_pipeline.py <decomposition_dir>
    python scripts/run_step_pipeline.py output/level3/11_VGG16/ --verbose

Expects the decomposition to have a steps/ directory with:
    steps/step_N_name/
        original.py              # or symlink to parent
        refactored.py            # parent rewritten with child calls
        children/                # child component files
        [weight_map.json]        # optional explicit weight map

Produces:
    verification/step_verification_summary.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def find_steps(decomp_dir: Path) -> List[Path]:
    """Find all step directories in the steps/ folder."""
    steps_dir = decomp_dir / "steps"
    if not steps_dir.exists():
        return []

    steps = []
    for step_dir in sorted(steps_dir.iterdir()):
        if step_dir.is_dir() and step_dir.name.startswith("step_"):
            steps.append(step_dir)

    return steps


def validate_step_dir(step_dir: Path) -> Dict:
    """Check that a step directory has the required files."""
    status = {
        "step_name": step_dir.name,
        "has_original": (step_dir / "original.py").exists(),
        "has_refactored": (step_dir / "refactored.py").exists(),
        "has_children": (step_dir / "children").exists(),
        "has_weight_map": (step_dir / "weight_map.json").exists(),
    }
    status["valid"] = status["has_original"] and status["has_refactored"]
    return status


def run_verify_step(
    step_dir: Path,
    scripts_dir: Path,
    verbose: bool = False,
) -> Dict:
    """Run verify_step.py for a single step."""
    original = step_dir / "original.py"
    refactored = step_dir / "refactored.py"
    output = step_dir / "verification_result.json"

    cmd = [
        sys.executable,
        str(scripts_dir / "verify_step.py"),
        "--original", str(original),
        "--refactored", str(refactored),
        "--output", str(output),
    ]

    # Add weight map if it exists
    weight_map = step_dir / "weight_map.json"
    if weight_map.exists():
        cmd.extend(["--weight-map", str(weight_map)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )

        if verbose:
            if result.stdout:
                # Indent output
                for line in result.stdout.strip().split("\n"):
                    print(f"    {line}")
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"    [stderr] {line}")

        # Read the result JSON if it was written
        if output.exists():
            with open(output, encoding="utf-8") as f:
                return json.load(f)

        return {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "exit_code": result.returncode,
            "stdout_snippet": result.stdout[:500] if result.stdout else "",
            "stderr_snippet": result.stderr[:500] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def run_extract_ops(
    step_dir: Path,
    scripts_dir: Path,
    verbose: bool = False,
) -> Dict:
    """Run extract_ops.py for a single step."""
    original = step_dir / "original.py"
    children_dir = step_dir / "children"
    output = step_dir / "coverage_report.json"

    if not children_dir.exists():
        return {"status": "SKIPPED", "reason": "No children directory"}

    # Find child files
    child_files = sorted(children_dir.glob("*.py"))
    if not child_files:
        return {"status": "SKIPPED", "reason": "No child files in children/"}

    cmd = [
        sys.executable,
        str(scripts_dir / "extract_ops.py"),
        "--model", str(original),
        "--children",
    ] + [str(f) for f in child_files] + [
        "--output", str(output),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if verbose and result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")

        if output.exists():
            with open(output, encoding="utf-8") as f:
                return json.load(f)

        return {
            "status": "DONE" if result.returncode == 0 else "PARTIAL",
            "exit_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def run_pipeline(
    decomp_dir: Path,
    verbose: bool = False,
) -> Dict:
    """Run the full step verification pipeline."""
    scripts_dir = Path(__file__).parent

    print("=" * 60)
    print("STEP VERIFICATION PIPELINE")
    print("=" * 60)
    print(f"Decomposition: {decomp_dir}")
    print()

    # Find all steps
    steps = find_steps(decomp_dir)
    if not steps:
        print("[WARN] No steps/ directory found or no step_* subdirectories.")
        print("       The agent may not have used step-by-step decomposition.")
        return {
            "timestamp": datetime.now().isoformat(),
            "decomp_dir": str(decomp_dir),
            "status": "NO_STEPS",
            "total_steps": 0,
            "steps": [],
        }

    print(f"Found {len(steps)} step(s)")
    print()

    # Validate and run each step
    step_results = []
    passed = 0
    failed = 0
    skipped = 0

    for i, step_dir in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {step_dir.name}")

        # Validate
        validation = validate_step_dir(step_dir)
        if not validation["valid"]:
            print(f"  SKIP: Missing required files")
            if not validation["has_original"]:
                print(f"    Missing: original.py")
            if not validation["has_refactored"]:
                print(f"    Missing: refactored.py")
            step_results.append({
                "step_name": step_dir.name,
                "status": "SKIPPED",
                "reason": "Missing required files",
                "validation": validation,
            })
            skipped += 1
            continue

        # Run verify_step
        print(f"  Running verify_step.py...")
        verify_result = run_verify_step(step_dir, scripts_dir, verbose)
        verify_status = verify_result.get("status", "UNKNOWN")

        # Run extract_ops
        print(f"  Running extract_ops.py...")
        coverage_result = run_extract_ops(step_dir, scripts_dir, verbose)
        coverage_pct = None
        if isinstance(coverage_result, dict) and "coverage" in coverage_result:
            coverage_pct = coverage_result["coverage"].get("coverage_pct")

        # Summarize
        step_summary = {
            "step_name": step_dir.name,
            "verification_status": verify_status,
            "coverage_pct": coverage_pct,
            "max_diff": verify_result.get("numerical_comparison", {}).get(
                "max_diff_across_trials"
            ),
            "anticheat_status": verify_result.get("anticheat", {}).get("status"),
        }

        if verify_status == "PASS":
            passed += 1
            status_str = "PASS"
        elif verify_status in ("FAIL", "ERROR", "TIMEOUT"):
            failed += 1
            status_str = "FAIL"
        else:
            skipped += 1
            status_str = verify_status

        coverage_str = f", coverage={coverage_pct}%" if coverage_pct is not None else ""
        print(f"  Result: {status_str}{coverage_str}")
        if step_summary.get("max_diff") is not None:
            print(f"  Max diff: {step_summary['max_diff']:.2e}")
        print()

        step_results.append(step_summary)

    # Summary
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total steps: {len(steps)}")
    print(f"Passed:      {passed}")
    print(f"Failed:      {failed}")
    print(f"Skipped:     {skipped}")

    overall_status = "PASS" if failed == 0 and passed > 0 else "FAIL"
    if passed == 0 and skipped == len(steps):
        overall_status = "NO_VALID_STEPS"

    print(f"\nOverall: {overall_status}")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "decomp_dir": str(decomp_dir),
        "status": overall_status,
        "total_steps": len(steps),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "steps": step_results,
    }

    # Write summary
    verification_dir = decomp_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    summary_path = verification_dir / "step_verification_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run step-by-step verification pipeline on a decomposition"
    )
    parser.add_argument(
        "decomp_dir",
        type=Path,
        help="Path to decomposition output directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output from each step",
    )

    args = parser.parse_args()

    if not args.decomp_dir.exists():
        print(f"Error: Directory not found: {args.decomp_dir}")
        sys.exit(2)

    result = run_pipeline(args.decomp_dir, args.verbose)

    if result["status"] == "PASS":
        sys.exit(0)
    elif result["status"] == "NO_STEPS":
        sys.exit(0)  # Not an error, just no steps to validate
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
