"""
Composition Test for 18_SqueezeNet Decomposition
Verifies that all decomposed components produce correct output shapes
and that the hierarchical decomposition is consistent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import importlib.util
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(relative_path):
    """Load a Python module from a relative path."""
    full_path = os.path.join(BASE_DIR, relative_path)
    spec = importlib.util.spec_from_file_location("module", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component(relative_path):
    """Test a single component by running its run_tests() function."""
    try:
        module = load_module(relative_path)
        result = module.run_tests()
        return result
    except Exception as e:
        print(f"  ERROR loading {relative_path}: {e}")
        return False


def test_all_level0_kernels():
    """Test all Level 0 kernel components."""
    print("=" * 60)
    print("LEVEL 0: Individual Kernels")
    print("=" * 60)
    kernels = [
        "level_0_kernel/conv2d_3x96_k7s2.py",
        "level_0_kernel/relu_96x253x253.py",
        "level_0_kernel/maxpool2d_96x253x253.py",
        "level_0_kernel/conv2d_96x16_k1.py",
        "level_0_kernel/relu_16x126x126.py",
        "level_0_kernel/conv2d_16x64_k1.py",
        "level_0_kernel/relu_64x126x126.py",
        "level_0_kernel/conv2d_16x64_k3p1.py",
        "level_0_kernel/cat_64_64_126.py",
        "level_0_kernel/conv2d_128x16_k1.py",
        "level_0_kernel/conv2d_128x32_k1.py",
        "level_0_kernel/relu_32x126x126.py",
        "level_0_kernel/conv2d_32x128_k1_126.py",
        "level_0_kernel/relu_128x126x126.py",
        "level_0_kernel/conv2d_32x128_k3p1_126.py",
        "level_0_kernel/cat_128_128_126.py",
        "level_0_kernel/maxpool2d_256x126x126.py",
        "level_0_kernel/conv2d_256x32_k1.py",
        "level_0_kernel/relu_32x63x63.py",
        "level_0_kernel/conv2d_32x128_k1_63.py",
        "level_0_kernel/relu_128x63x63.py",
        "level_0_kernel/conv2d_32x128_k3p1_63.py",
        "level_0_kernel/cat_128_128_63.py",
        "level_0_kernel/conv2d_256x48_k1.py",
        "level_0_kernel/relu_48x63x63.py",
        "level_0_kernel/conv2d_48x192_k1.py",
        "level_0_kernel/relu_192x63x63.py",
        "level_0_kernel/conv2d_48x192_k3p1.py",
        "level_0_kernel/cat_192_192_63.py",
        "level_0_kernel/conv2d_384x48_k1.py",
        "level_0_kernel/conv2d_384x64_k1.py",
        "level_0_kernel/relu_64x63x63.py",
        "level_0_kernel/conv2d_64x256_k1_63.py",
        "level_0_kernel/relu_256x63x63.py",
        "level_0_kernel/conv2d_64x256_k3p1_63.py",
        "level_0_kernel/cat_256_256_63.py",
        "level_0_kernel/maxpool2d_512x63x63.py",
        "level_0_kernel/conv2d_512x64_k1.py",
        "level_0_kernel/relu_64x31x31.py",
        "level_0_kernel/conv2d_64x256_k1_31.py",
        "level_0_kernel/relu_256x31x31.py",
        "level_0_kernel/conv2d_64x256_k3p1_31.py",
        "level_0_kernel/cat_256_256_31.py",
        "level_0_kernel/dropout_512x31x31.py",
        "level_0_kernel/conv2d_512x1000_k1.py",
        "level_0_kernel/relu_1000x31x31.py",
        "level_0_kernel/adaptive_avg_pool2d_1000.py",
        "level_0_kernel/flatten_1000x1x1.py",
    ]
    passed = 0
    failed = 0
    for k in kernels:
        print(f"  Testing {k}...", end=" ")
        if test_component(k):
            passed += 1
        else:
            failed += 1
    print(f"\nLevel 0 Results: {passed}/{passed + failed} passed")
    return failed == 0


def test_all_level1_fusions():
    """Test all Level 1 fusion components."""
    print("\n" + "=" * 60)
    print("LEVEL 1: Fusions")
    print("=" * 60)
    fusions = [
        "level_1_fusion/squeeze_conv_relu_96x16.py",
        "level_1_fusion/expand_concat_16x64x64.py",
        "level_1_fusion/squeeze_conv_relu_128x16.py",
        "level_1_fusion/expand_concat_16x64x64_2.py",
        "level_1_fusion/squeeze_conv_relu_128x32.py",
        "level_1_fusion/expand_concat_32x128x128.py",
        "level_1_fusion/squeeze_conv_relu_256x32.py",
        "level_1_fusion/expand_concat_32x128x128_2.py",
        "level_1_fusion/squeeze_conv_relu_256x48.py",
        "level_1_fusion/expand_concat_48x192x192.py",
        "level_1_fusion/squeeze_conv_relu_384x48.py",
        "level_1_fusion/expand_concat_48x192x192_2.py",
        "level_1_fusion/squeeze_conv_relu_384x64.py",
        "level_1_fusion/expand_concat_64x256x256.py",
        "level_1_fusion/squeeze_conv_relu_512x64.py",
        "level_1_fusion/expand_concat_64x256x256_2.py",
    ]
    passed = 0
    failed = 0
    for f in fusions:
        print(f"  Testing {f}...", end=" ")
        if test_component(f):
            passed += 1
        else:
            failed += 1
    print(f"\nLevel 1 Results: {passed}/{passed + failed} passed")
    return failed == 0


def test_all_level2_layers():
    """Test all Level 2 layer components."""
    print("\n" + "=" * 60)
    print("LEVEL 2: Layers")
    print("=" * 60)
    layers = [
        "level_2_layer/initial_conv_block.py",
        "level_2_layer/fire_module_1.py",
        "level_2_layer/fire_module_2.py",
        "level_2_layer/fire_module_3.py",
        "level_2_layer/fire_module_4.py",
        "level_2_layer/fire_module_5.py",
        "level_2_layer/fire_module_6.py",
        "level_2_layer/fire_module_7.py",
        "level_2_layer/fire_module_8.py",
        "level_2_layer/classifier.py",
    ]
    passed = 0
    failed = 0
    for l in layers:
        print(f"  Testing {l}...", end=" ")
        if test_component(l):
            passed += 1
        else:
            failed += 1
    print(f"\nLevel 2 Results: {passed}/{passed + failed} passed")
    return failed == 0


def test_level3_model():
    """Test the full Level 3 model."""
    print("\n" + "=" * 60)
    print("LEVEL 3: Full Model")
    print("=" * 60)
    print(f"  Testing level_3_model/squeezenet.py...", end=" ")
    result = test_component("level_3_model/squeezenet.py")
    print(f"\nLevel 3 Results: {'1/1 passed' if result else '0/1 passed'}")
    return result


def test_data_flow_consistency():
    """Verify that the data flow between layers is consistent."""
    print("\n" + "=" * 60)
    print("DATA FLOW CONSISTENCY CHECK")
    print("=" * 60)

    all_pass = True

    # Verify sequential data flow matches expected shapes
    flow = [
        ("level_2_layer/initial_conv_block.py", (64, 3, 512, 512), (64, 96, 126, 126)),
        ("level_2_layer/fire_module_1.py", (64, 96, 126, 126), (64, 128, 126, 126)),
        ("level_2_layer/fire_module_2.py", (64, 128, 126, 126), (64, 128, 126, 126)),
        ("level_2_layer/fire_module_3.py", (64, 128, 126, 126), (64, 256, 126, 126)),
        ("level_0_kernel/maxpool2d_256x126x126.py", (64, 256, 126, 126), (64, 256, 63, 63)),
        ("level_2_layer/fire_module_4.py", (64, 256, 63, 63), (64, 256, 63, 63)),
        ("level_2_layer/fire_module_5.py", (64, 256, 63, 63), (64, 384, 63, 63)),
        ("level_2_layer/fire_module_6.py", (64, 384, 63, 63), (64, 384, 63, 63)),
        ("level_2_layer/fire_module_7.py", (64, 384, 63, 63), (64, 512, 63, 63)),
        ("level_0_kernel/maxpool2d_512x63x63.py", (64, 512, 63, 63), (64, 512, 31, 31)),
        ("level_2_layer/fire_module_8.py", (64, 512, 31, 31), (64, 512, 31, 31)),
        ("level_2_layer/classifier.py", (64, 512, 31, 31), (64, 1000)),
    ]

    for component, expected_in, expected_out in flow:
        try:
            module = load_module(component)
            inputs = module.get_inputs()
            in_shape = tuple(inputs[0].shape)
            expected_out_shapes = module.get_expected_output_shape()

            if in_shape != expected_in:
                print(f"  FAIL {component}: input shape {in_shape} != expected {expected_in}")
                all_pass = False
            elif tuple(expected_out_shapes[0]) != expected_out:
                print(f"  FAIL {component}: output shape {expected_out_shapes[0]} != expected {expected_out}")
                all_pass = False
            else:
                print(f"  OK {component}: {expected_in} -> {expected_out}")
        except Exception as e:
            print(f"  ERROR {component}: {e}")
            all_pass = False

    # Verify output of one layer matches input of next
    print("\n  Checking layer-to-layer shape consistency...")
    for i in range(len(flow) - 1):
        _, _, out_shape = flow[i]
        _, in_shape, _ = flow[i + 1]
        if out_shape != in_shape:
            print(f"  MISMATCH: {flow[i][0]} output {out_shape} != {flow[i+1][0]} input {in_shape}")
            all_pass = False

    if all_pass:
        print("  All data flow checks PASSED")
    return all_pass


def test_decomposition_tree():
    """Verify the decomposition tree JSON is valid and consistent."""
    print("\n" + "=" * 60)
    print("DECOMPOSITION TREE VALIDATION")
    print("=" * 60)

    tree_path = os.path.join(BASE_DIR, "decomposition_tree.json")
    try:
        with open(tree_path, "r") as f:
            tree = json.load(f)
        print("  JSON is valid")

        # Check all referenced files exist
        all_exist = True
        for level_name, level_data in tree["levels"].items():
            for component_name, component_data in level_data.items():
                component_path = os.path.join(BASE_DIR, level_name, component_name)
                if not os.path.exists(component_path):
                    print(f"  MISSING: {level_name}/{component_name}")
                    all_exist = False
                if "children" in component_data:
                    for child in component_data["children"]:
                        child_path = os.path.join(BASE_DIR, child)
                        if not os.path.exists(child_path):
                            print(f"  MISSING child: {child} (referenced by {level_name}/{component_name})")
                            all_exist = False

        if all_exist:
            print("  All referenced files exist")
        return all_exist
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    print("SqueezeNet Hierarchical Decomposition - Composition Test")
    print("=" * 60)

    results = {}

    results["decomposition_tree"] = test_decomposition_tree()
    results["data_flow"] = test_data_flow_consistency()
    results["level0"] = test_all_level0_kernels()
    results["level1"] = test_all_level1_fusions()
    results["level2"] = test_all_level2_layers()
    results["level3"] = test_level3_model()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_pass = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
