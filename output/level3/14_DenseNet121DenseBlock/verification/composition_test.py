"""
Composition Verification Test for 14_DenseNet121DenseBlock
Source: data/kernelbench/level3/14_DenseNet121DenseBlock.py

This test verifies that the decomposition is correct by:
1. Running each component individually and checking shapes
2. Verifying that composing sub-components produces the same output as the parent
3. Checking numerical equivalence between levels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import importlib.util


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(relative_path):
    """Load a Python module from a relative path within the decomposition."""
    full_path = os.path.join(BASE_DIR, relative_path)
    spec = importlib.util.spec_from_file_location("module", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_level_0_kernels():
    """Test all Level 0 kernel components individually."""
    print("=" * 60)
    print("Testing Level 0 Kernels")
    print("=" * 60)
    kernel_files = [
        "level_0_kernel/batch_norm2d_32.py",
        "level_0_kernel/relu_32x224x224.py",
        "level_0_kernel/conv2d_32x32_k3p1.py",
        "level_0_kernel/dropout_32x224x224.py",
        "level_0_kernel/cat_32_32.py",
        "level_0_kernel/batch_norm2d_64.py",
        "level_0_kernel/relu_64x224x224.py",
        "level_0_kernel/conv2d_64x32_k3p1.py",
        "level_0_kernel/dropout_64x224x224.py",
        "level_0_kernel/cat_64_32.py",
        "level_0_kernel/batch_norm2d_96.py",
        "level_0_kernel/relu_96x224x224.py",
        "level_0_kernel/conv2d_96x32_k3p1.py",
        "level_0_kernel/cat_96_32.py",
        "level_0_kernel/batch_norm2d_128.py",
        "level_0_kernel/relu_128x224x224.py",
        "level_0_kernel/conv2d_128x32_k3p1.py",
        "level_0_kernel/cat_128_32.py",
        "level_0_kernel/batch_norm2d_160.py",
        "level_0_kernel/relu_160x224x224.py",
        "level_0_kernel/conv2d_160x32_k3p1.py",
        "level_0_kernel/cat_160_32.py",
        "level_0_kernel/batch_norm2d_192.py",
        "level_0_kernel/relu_192x224x224.py",
        "level_0_kernel/conv2d_192x32_k3p1.py",
        "level_0_kernel/cat_192_32.py",
    ]
    all_pass = True
    for kf in kernel_files:
        mod = load_module(kf)
        result = mod.run_tests()
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {kf}")
        if not result:
            all_pass = False
    return all_pass


def test_level_1_fusions():
    """Test all Level 1 fusion components individually."""
    print("=" * 60)
    print("Testing Level 1 Fusions")
    print("=" * 60)
    fusion_files = [
        "level_1_fusion/bn_relu_conv_drop_32.py",
        "level_1_fusion/bn_relu_conv_drop_64.py",
        "level_1_fusion/bn_relu_conv_drop_96.py",
        "level_1_fusion/bn_relu_conv_drop_128.py",
        "level_1_fusion/bn_relu_conv_drop_160.py",
        "level_1_fusion/bn_relu_conv_drop_192.py",
    ]
    all_pass = True
    for ff in fusion_files:
        mod = load_module(ff)
        result = mod.run_tests()
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {ff}")
        if not result:
            all_pass = False
    return all_pass


def test_level_2_layers():
    """Test all Level 2 layer components individually."""
    print("=" * 60)
    print("Testing Level 2 Layers")
    print("=" * 60)
    layer_files = [
        "level_2_layer/dense_layer_0.py",
        "level_2_layer/dense_layer_1.py",
        "level_2_layer/dense_layer_2.py",
        "level_2_layer/dense_layer_3.py",
        "level_2_layer/dense_layer_4.py",
        "level_2_layer/dense_layer_5.py",
    ]
    all_pass = True
    for lf in layer_files:
        mod = load_module(lf)
        result = mod.run_tests()
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {lf}")
        if not result:
            all_pass = False
    return all_pass


def test_level_3_model():
    """Test the Level 3 model component."""
    print("=" * 60)
    print("Testing Level 3 Model")
    print("=" * 60)
    mod = load_module("level_3_model/dense_block.py")
    result = mod.run_tests()
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] level_3_model/dense_block.py")
    return result


def test_fusion_composition():
    """
    Verify that composing L0 kernels matches L1 fusion output.
    Test for the first fusion (BN(32)+ReLU+Conv(32->32)+Drop).
    """
    print("=" * 60)
    print("Testing Fusion Composition (L0 -> L1)")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(10, 32, 224, 224)

    # Load L0 kernels
    bn_mod = load_module("level_0_kernel/batch_norm2d_32.py")
    relu_mod = load_module("level_0_kernel/relu_32x224x224.py")
    conv_mod = load_module("level_0_kernel/conv2d_32x32_k3p1.py")
    drop_mod = load_module("level_0_kernel/dropout_32x224x224.py")

    bn_model = bn_mod.Model()
    relu_model = relu_mod.Model()
    conv_model = conv_mod.Model()
    drop_model = drop_mod.Model()

    # Load L1 fusion
    fusion_mod = load_module("level_1_fusion/bn_relu_conv_drop_32.py")
    fusion_model = fusion_mod.Model()

    # Copy weights from fusion to kernels
    bn_model.bn.load_state_dict(fusion_model.bn.state_dict())
    conv_model.conv.load_state_dict(fusion_model.conv.state_dict())

    bn_model.eval()
    relu_model.eval()
    conv_model.eval()
    drop_model.eval()
    fusion_model.eval()

    with torch.no_grad():
        # Composed L0 path
        out_bn = bn_model(x)
        out_relu = relu_model(out_bn)
        out_conv = conv_model(out_relu)
        out_composed = drop_model(out_conv)

        # Direct L1 path
        out_fusion = fusion_model(x)

    match = torch.allclose(out_composed, out_fusion, atol=1e-6)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] L0 kernels composed == L1 fusion (bn_relu_conv_drop_32)")
    if not match:
        max_diff = (out_composed - out_fusion).abs().max().item()
        print(f"    Max difference: {max_diff}")
    return match


def test_layer_composition():
    """
    Verify that composing L1 fusion + cat matches L2 layer output.
    Test for dense_layer_0.
    """
    print("=" * 60)
    print("Testing Layer Composition (L1 -> L2)")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(10, 32, 224, 224)

    # Load L1 fusion and L0 cat
    fusion_mod = load_module("level_1_fusion/bn_relu_conv_drop_32.py")
    cat_mod = load_module("level_0_kernel/cat_32_32.py")

    fusion_model = fusion_mod.Model()
    cat_model = cat_mod.Model()

    # Load L2 layer
    layer_mod = load_module("level_2_layer/dense_layer_0.py")
    layer_model = layer_mod.Model()

    # Copy weights from layer to fusion
    fusion_model.bn.load_state_dict(layer_model.bn.state_dict())
    fusion_model.conv.load_state_dict(layer_model.conv.state_dict())

    fusion_model.eval()
    cat_model.eval()
    layer_model.eval()

    with torch.no_grad():
        # Composed path: fusion then cat
        new_feature = fusion_model(x)
        out_composed = cat_model(x, new_feature)

        # Direct L2 path
        out_layer = layer_model(x)

    match = torch.allclose(out_composed, out_layer, atol=1e-6)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] L1 fusion + cat composed == L2 layer (dense_layer_0)")
    if not match:
        max_diff = (out_composed - out_layer).abs().max().item()
        print(f"    Max difference: {max_diff}")
    return match


def test_model_composition():
    """
    Verify that composing L2 layers sequentially matches L3 model output.
    """
    print("=" * 60)
    print("Testing Model Composition (L2 -> L3)")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(10, 32, 224, 224)

    # Load L3 model
    model_mod = load_module("level_3_model/dense_block.py")
    full_model = model_mod.Model(*model_mod.get_init_inputs())
    full_model.eval()

    # Load all L2 layers
    layer_mods = []
    for i in range(6):
        mod = load_module(f"level_2_layer/dense_layer_{i}.py")
        layer_mods.append(mod)

    layer_models = [mod.Model() for mod in layer_mods]

    # Copy weights from full model to individual layers
    for i, layer_model in enumerate(layer_models):
        layer_model.bn.load_state_dict(full_model.layers[i][0].state_dict())
        layer_model.conv.load_state_dict(full_model.layers[i][2].state_dict())
        layer_model.eval()

    with torch.no_grad():
        # Composed L2 path
        current = x.clone()
        features = [x.clone()]
        for i, layer_model in enumerate(layer_models):
            new_feature = layer_model.drop(layer_model.conv(layer_model.relu(layer_model.bn(current))))
            features.append(new_feature)
            current = torch.cat(features, dim=1)
        out_composed = current

        # Direct L3 path
        out_model = full_model(x)

    match = torch.allclose(out_composed, out_model, atol=1e-5)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] L2 layers composed == L3 model (dense_block)")
    if not match:
        max_diff = (out_composed - out_model).abs().max().item()
        print(f"    Max difference: {max_diff}")
    return match


def test_end_to_end_shapes():
    """Verify the full data flow shapes through all 6 layers."""
    print("=" * 60)
    print("Testing End-to-End Shape Flow")
    print("=" * 60)

    expected_shapes = [
        (10, 32, 224, 224),   # Input
        (10, 64, 224, 224),   # After layer 0
        (10, 96, 224, 224),   # After layer 1
        (10, 128, 224, 224),  # After layer 2
        (10, 160, 224, 224),  # After layer 3
        (10, 192, 224, 224),  # After layer 4
        (10, 224, 224, 224),  # After layer 5 (final output)
    ]

    model_mod = load_module("level_3_model/dense_block.py")
    model = model_mod.Model(*model_mod.get_init_inputs())
    model.eval()

    x = torch.randn(10, 32, 224, 224)
    all_pass = True

    with torch.no_grad():
        features = [x]
        current = x
        for i, layer in enumerate(model.layers):
            new_feature = layer(current)
            features.append(new_feature)
            current = torch.cat(features, 1)
            actual_shape = tuple(current.shape)
            expected = expected_shapes[i + 1]
            match = actual_shape == expected
            status = "PASS" if match else "FAIL"
            print(f"  [{status}] After layer {i}: {actual_shape} (expected {expected})")
            if not match:
                all_pass = False

    return all_pass


def main():
    results = {}

    results["Level 0 Kernels"] = test_level_0_kernels()
    print()
    results["Level 1 Fusions"] = test_level_1_fusions()
    print()
    results["Level 2 Layers"] = test_level_2_layers()
    print()
    results["Level 3 Model"] = test_level_3_model()
    print()
    results["Fusion Composition (L0->L1)"] = test_fusion_composition()
    print()
    results["Layer Composition (L1->L2)"] = test_layer_composition()
    print()
    results["Model Composition (L2->L3)"] = test_model_composition()
    print()
    results["End-to-End Shapes"] = test_end_to_end_shapes()
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if not result:
            all_pass = False

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
