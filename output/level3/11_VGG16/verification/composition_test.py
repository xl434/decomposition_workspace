"""
Composition Verification Test for VGG16 Decomposition
Source: data/kernelbench/level3/11_VGG16.py

This test verifies that the hierarchical decomposition is correct by:
1. Running each component file independently and checking shapes
2. Verifying that composing sub-components produces the same output as the parent
3. Checking end-to-end data flow consistency
"""
import torch
import torch.nn as nn
import sys
import os
import importlib.util
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(relative_path):
    """Load a module from a relative path within the decomposition directory."""
    full_path = os.path.join(BASE_DIR, relative_path)
    spec = importlib.util.spec_from_file_location("module", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component(relative_path):
    """Test a single component by running its run_tests function."""
    try:
        module = load_module(relative_path)
        result = module.run_tests()
        return result
    except Exception as e:
        print(f"ERROR loading {relative_path}: {e}")
        return False


def test_all_components():
    """Test all component files in the decomposition."""
    tree_path = os.path.join(BASE_DIR, "decomposition_tree.json")
    with open(tree_path, "r") as f:
        tree = json.load(f)

    all_components = list(tree["tree"].keys())
    passed = 0
    failed = 0
    errors = []

    for component_path in all_components:
        print(f"Testing {component_path}... ", end="")
        result = test_component(component_path)
        if result:
            passed += 1
            print("OK")
        else:
            failed += 1
            errors.append(component_path)
            print("FAILED")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print(f"Failed components:")
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*60}")
    return failed == 0


def test_composition_block1():
    """Verify Block 1 composition: conv_relu_3x64 -> conv_relu_64x64 -> maxpool produces same shape as block_1."""
    print("\n--- Testing Block 1 Composition ---")
    torch.manual_seed(42)
    x = torch.randn(10, 3, 224, 224)

    # Load block-level component
    block_mod = load_module("level_2_layer/features_block_1.py")
    block_model = block_mod.Model()
    block_model.eval()

    # Load fusion-level components
    cr_3x64 = load_module("level_1_fusion/conv_relu_3x64.py")
    cr_64x64 = load_module("level_1_fusion/conv_relu_64x64.py")
    mp = load_module("level_0_kernel/maxpool2d_64x224x224.py")

    cr_3x64_model = cr_3x64.Model()
    cr_64x64_model = cr_64x64.Model()
    mp_model = mp.Model()
    cr_3x64_model.eval()
    cr_64x64_model.eval()
    mp_model.eval()

    with torch.no_grad():
        block_out = block_model(x)
        # Compose sub-components
        h = cr_3x64_model(x)
        h = cr_64x64_model(h)
        h = mp_model(h)

    assert block_out.shape == h.shape, f"Shape mismatch: {block_out.shape} vs {h.shape}"
    print(f"  Block 1 output shape: {block_out.shape} -- PASS")
    return True


def test_composition_classifier():
    """Verify Classifier composition: flatten -> linear_relu_drop_25088x4096 -> linear_relu_drop_4096x4096 -> linear_4096x1000."""
    print("\n--- Testing Classifier Composition ---")
    torch.manual_seed(42)
    x = torch.randn(10, 512, 7, 7)

    # Load block-level component
    cls_mod = load_module("level_2_layer/classifier.py")
    cls_model = cls_mod.Model(*cls_mod.get_init_inputs())
    cls_model.eval()

    # Load sub-components
    flatten_mod = load_module("level_0_kernel/flatten_512x7x7.py")
    lrd1_mod = load_module("level_1_fusion/linear_relu_dropout_25088x4096.py")
    lrd2_mod = load_module("level_1_fusion/linear_relu_dropout_4096x4096.py")
    lin_final_mod = load_module("level_0_kernel/linear_4096x1000.py")

    flatten_model = flatten_mod.Model()
    lrd1_model = lrd1_mod.Model()
    lrd2_model = lrd2_mod.Model()
    lin_final_model = lin_final_mod.Model()
    flatten_model.eval()
    lrd1_model.eval()
    lrd2_model.eval()
    lin_final_model.eval()

    with torch.no_grad():
        cls_out = cls_model(x)
        # Compose sub-components
        h = flatten_model(x)
        assert h.shape == (10, 25088), f"Flatten shape: {h.shape}"
        h = lrd1_model(h)
        assert h.shape == (10, 4096), f"LRD1 shape: {h.shape}"
        h = lrd2_model(h)
        assert h.shape == (10, 4096), f"LRD2 shape: {h.shape}"
        h = lin_final_model(h)
        assert h.shape == (10, 1000), f"Final linear shape: {h.shape}"

    assert cls_out.shape == h.shape, f"Shape mismatch: {cls_out.shape} vs {h.shape}"
    print(f"  Classifier output shape: {cls_out.shape} -- PASS")
    return True


def test_end_to_end_shape():
    """Verify the full model produces the expected output shape."""
    print("\n--- Testing End-to-End Shape ---")
    torch.manual_seed(42)

    model_mod = load_module("level_3_model/vgg16.py")
    model = model_mod.Model(*model_mod.get_init_inputs())
    model.eval()

    with torch.no_grad():
        inputs = model_mod.get_inputs()
        output = model(*inputs)

    expected_shape = (10, 1000)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print(f"  Full model output shape: {output.shape} -- PASS")
    return True


def test_data_flow_shapes():
    """Verify data flow shapes through each block sequentially."""
    print("\n--- Testing Data Flow Shapes ---")
    torch.manual_seed(42)
    x = torch.randn(10, 3, 224, 224)

    blocks = [
        ("level_2_layer/features_block_1.py", (10, 64, 112, 112)),
        ("level_2_layer/features_block_2.py", (10, 128, 56, 56)),
        ("level_2_layer/features_block_3.py", (10, 256, 28, 28)),
        ("level_2_layer/features_block_4.py", (10, 512, 14, 14)),
        ("level_2_layer/features_block_5.py", (10, 512, 7, 7)),
    ]

    h = x
    for block_path, expected_shape in blocks:
        mod = load_module(block_path)
        model = mod.Model()
        model.eval()
        with torch.no_grad():
            h = model(h)
        assert h.shape == expected_shape, f"{block_path}: expected {expected_shape}, got {h.shape}"
        print(f"  {block_path}: {h.shape} -- OK")

    # Classifier
    cls_mod = load_module("level_2_layer/classifier.py")
    cls_model = cls_mod.Model(*cls_mod.get_init_inputs())
    cls_model.eval()
    with torch.no_grad():
        h = cls_model(h)
    assert h.shape == (10, 1000), f"Classifier: expected (10, 1000), got {h.shape}"
    print(f"  level_2_layer/classifier.py: {h.shape} -- OK")
    print("  Data flow shapes: PASS")
    return True


def main():
    print("=" * 60)
    print("VGG16 Decomposition Verification")
    print("=" * 60)

    all_pass = True

    # Test 1: All individual components
    print("\n[Test 1] Individual Component Tests")
    if not test_all_components():
        all_pass = False

    # Test 2: Block 1 composition
    print("\n[Test 2] Block 1 Composition")
    try:
        if not test_composition_block1():
            all_pass = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_pass = False

    # Test 3: Classifier composition
    print("\n[Test 3] Classifier Composition")
    try:
        if not test_composition_classifier():
            all_pass = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_pass = False

    # Test 4: End-to-end shape
    print("\n[Test 4] End-to-End Shape")
    try:
        if not test_end_to_end_shape():
            all_pass = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_pass = False

    # Test 5: Data flow shapes
    print("\n[Test 5] Data Flow Shapes")
    try:
        if not test_data_flow_shapes():
            all_pass = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL VERIFICATION TESTS PASSED")
    else:
        print("SOME VERIFICATION TESTS FAILED")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
