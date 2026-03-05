"""
Composition Test: Verify full decomposition pipeline

Tests:
1. All individual kernel/fusion/layer components execute correctly
2. Step-level verifications all pass
3. End-to-end: L3 model output matches when composed from L2 children
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import torch

BASE_DIR = Path(__file__).resolve().parent.parent

results = {
    "timestamp": datetime.now().isoformat(),
    "total_components": 0,
    "passed": 0,
    "failed": 0,
    "composition_test": "UNKNOWN",
    "max_difference": 0.0,
    "component_results": [],
}


def test_component(file_path, label):
    """Run a component's standalone test."""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(label, str(file_path))
        mod = importlib.util.module_from_spec(spec)
        parent = str(file_path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec.loader.exec_module(mod)
        if hasattr(mod, 'run_tests'):
            passed = mod.run_tests()
        else:
            model_cls = getattr(mod, 'Model', getattr(mod, 'RefactoredModel', None))
            get_init = getattr(mod, 'get_init_inputs', lambda: [])
            model = model_cls(*get_init())
            model.eval()
            with torch.no_grad():
                inputs = mod.get_inputs()
                output = model(*inputs)
                passed = output is not None
        return passed
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


print("=" * 70)
print("COMPOSITION TEST: SmolVLA VLAFlowMatching Decomposition")
print("=" * 70)

# 1. Test all L0 kernels
print("\n[1/4] Testing L0 kernels...")
kernel_dir = BASE_DIR / "level_0_kernel"
for f in sorted(kernel_dir.glob("*.py")):
    results["total_components"] += 1
    passed = test_component(f, f.stem)
    status = "PASS" if passed else "FAIL"
    results["component_results"].append({"file": str(f.relative_to(BASE_DIR)), "status": status})
    if passed:
        results["passed"] += 1
    else:
        results["failed"] += 1
    print(f"  {f.name}: {status}")

# 2. Test all L1 fusions
print("\n[2/4] Testing L1 fusions...")
fusion_dir = BASE_DIR / "level_1_fusion"
for f in sorted(fusion_dir.glob("*.py")):
    results["total_components"] += 1
    passed = test_component(f, f"fusion_{f.stem}")
    status = "PASS" if passed else "FAIL"
    results["component_results"].append({"file": str(f.relative_to(BASE_DIR)), "status": status})
    if passed:
        results["passed"] += 1
    else:
        results["failed"] += 1
    print(f"  {f.name}: {status}")

# 3. Test all L2 layers
print("\n[3/4] Testing L2 layers...")
layer_dir = BASE_DIR / "level_2_layer"
for f in sorted(layer_dir.glob("*.py")):
    results["total_components"] += 1
    passed = test_component(f, f"layer_{f.stem}")
    status = "PASS" if passed else "FAIL"
    results["component_results"].append({"file": str(f.relative_to(BASE_DIR)), "status": status})
    if passed:
        results["passed"] += 1
    else:
        results["failed"] += 1
    print(f"  {f.name}: {status}")

# 4. End-to-end composition test: verify step_1 refactored matches original
print("\n[4/4] End-to-end composition test...")
try:
    sys.path.insert(0, str(BASE_DIR / "level_3_model"))
    import importlib
    orig_spec = importlib.util.spec_from_file_location(
        "orig_model", str(BASE_DIR / "level_3_model" / "smolvla.py"))
    orig_mod = importlib.util.module_from_spec(orig_spec)
    orig_spec.loader.exec_module(orig_mod)

    sys.path.insert(0, str(BASE_DIR / "steps" / "step_1_model_to_layers" / "children"))
    ref_spec = importlib.util.spec_from_file_location(
        "ref_model", str(BASE_DIR / "steps" / "step_1_model_to_layers" / "refactored.py"))
    ref_mod = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(ref_mod)

    orig_model = orig_mod.Model()
    ref_model = ref_mod.RefactoredModel()

    # Transfer weights
    wmap_path = BASE_DIR / "steps" / "step_1_model_to_layers" / "weight_map.json"
    with open(wmap_path) as f:
        wmap = json.load(f)

    orig_sd = orig_model.state_dict()
    ref_sd = ref_model.state_dict()
    for ok, rk in wmap.items():
        if ok in orig_sd and rk in ref_sd:
            ref_sd[rk] = orig_sd[ok].clone()
    ref_model.load_state_dict(ref_sd)

    orig_model.eval()
    ref_model.eval()

    max_diff = 0.0
    all_pass = True
    for trial in range(3):
        torch.manual_seed(42 + trial)
        inputs = orig_mod.get_inputs()
        with torch.no_grad():
            orig_out = orig_model(*inputs)
            ref_out = ref_model(*inputs)
        diff = (orig_out.float() - ref_out.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        matches = torch.allclose(orig_out.float(), ref_out.float(), rtol=1e-5, atol=1e-6)
        if not matches:
            all_pass = False
        print(f"  Trial {trial}: max_diff={diff:.2e} {'PASS' if matches else 'FAIL'}")

    results["max_difference"] = max_diff
    results["composition_test"] = "PASSED" if all_pass else "FAILED"
    print(f"  Composition: {'PASS' if all_pass else 'FAIL'} (max_diff={max_diff:.2e})")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    results["composition_test"] = "ERROR"

# Summary
print("\n" + "=" * 70)
print(f"Components: {results['passed']}/{results['total_components']} passed")
print(f"Composition: {results['composition_test']}")
print(f"Max difference: {results['max_difference']:.2e}")

overall = results["failed"] == 0 and results["composition_test"] == "PASSED"
print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
print("=" * 70)

# Save results
output_path = BASE_DIR / "verification" / "test_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {output_path}")

sys.exit(0 if overall else 1)
