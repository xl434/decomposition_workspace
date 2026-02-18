"""
Composition Verification Test for 29_SwinMLP
Source: data/kernelbench/level3/29_SwinMLP.py

Tests that all decomposed components produce correct output shapes
and that the hierarchical composition is consistent.
"""
import torch
import sys
import os
import importlib.util
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(relative_path):
    """Load a module from a relative path within the decomposition."""
    full_path = os.path.join(BASE_DIR, relative_path)
    spec = importlib.util.spec_from_file_location("module", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component(name, relative_path):
    """Test a single component by running its run_tests function."""
    try:
        module = load_module(relative_path)
        result = module.run_tests()
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        return result
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return False


def test_data_flow():
    """Test that data flows correctly through the decomposed pipeline."""
    print("\n=== Data Flow Verification ===")
    results = []

    try:
        # Test PatchEmbed
        print("  Testing PatchEmbed data flow...")
        patch_embed = load_module("level_2_layer/patch_embed.py")
        model_pe = patch_embed.Model()
        model_pe.eval()
        with torch.no_grad():
            x = torch.randn(10, 3, 224, 224)
            out_pe = model_pe(x)
            assert out_pe.shape == (10, 3136, 96), f"PatchEmbed output shape: {out_pe.shape}"
            print(f"    PatchEmbed: {x.shape} -> {out_pe.shape} OK")
            results.append(True)

        # Test BasicLayer Stage 0
        print("  Testing BasicLayer Stage 0 data flow...")
        layer0 = load_module("level_2_layer/basic_layer_0.py")
        model_l0 = layer0.Model()
        model_l0.eval()
        with torch.no_grad():
            x_l0 = torch.randn(10, 3136, 96)
            out_l0 = model_l0(x_l0)
            assert out_l0.shape == (10, 784, 192), f"Stage 0 output shape: {out_l0.shape}"
            print(f"    Stage 0: {x_l0.shape} -> {out_l0.shape} OK")
            results.append(True)

        # Test BasicLayer Stage 1
        print("  Testing BasicLayer Stage 1 data flow...")
        layer1 = load_module("level_2_layer/basic_layer_1.py")
        model_l1 = layer1.Model()
        model_l1.eval()
        with torch.no_grad():
            x_l1 = torch.randn(10, 784, 192)
            out_l1 = model_l1(x_l1)
            assert out_l1.shape == (10, 196, 384), f"Stage 1 output shape: {out_l1.shape}"
            print(f"    Stage 1: {x_l1.shape} -> {out_l1.shape} OK")
            results.append(True)

        # Test BasicLayer Stage 2
        print("  Testing BasicLayer Stage 2 data flow...")
        layer2 = load_module("level_2_layer/basic_layer_2.py")
        model_l2 = layer2.Model()
        model_l2.eval()
        with torch.no_grad():
            x_l2 = torch.randn(10, 196, 384)
            out_l2 = model_l2(x_l2)
            assert out_l2.shape == (10, 49, 768), f"Stage 2 output shape: {out_l2.shape}"
            print(f"    Stage 2: {x_l2.shape} -> {out_l2.shape} OK")
            results.append(True)

        # Test BasicLayer Stage 3
        print("  Testing BasicLayer Stage 3 data flow...")
        layer3 = load_module("level_2_layer/basic_layer_3.py")
        model_l3 = layer3.Model()
        model_l3.eval()
        with torch.no_grad():
            x_l3 = torch.randn(10, 49, 768)
            out_l3 = model_l3(x_l3)
            assert out_l3.shape == (10, 49, 768), f"Stage 3 output shape: {out_l3.shape}"
            print(f"    Stage 3: {x_l3.shape} -> {out_l3.shape} OK")
            results.append(True)

        # Test Head
        print("  Testing Head data flow...")
        head = load_module("level_2_layer/head.py")
        model_head = head.Model()
        model_head.eval()
        with torch.no_grad():
            x_head = torch.randn(10, 49, 768)
            out_head = model_head(x_head)
            assert out_head.shape == (10, 1000), f"Head output shape: {out_head.shape}"
            print(f"    Head: {x_head.shape} -> {out_head.shape} OK")
            results.append(True)

        # Test full model
        print("  Testing full model data flow...")
        full_model = load_module("level_3_model/swin_mlp.py")
        model_full = full_model.Model()
        model_full.eval()
        with torch.no_grad():
            x_full = torch.randn(10, 3, 224, 224)
            out_full = model_full(x_full)
            assert out_full.shape == (10, 1000), f"Full model output shape: {out_full.shape}"
            print(f"    Full model: {x_full.shape} -> {out_full.shape} OK")
            results.append(True)

    except Exception as e:
        print(f"    FAIL: {e}")
        traceback.print_exc()
        results.append(False)

    return all(results)


def main():
    print("=" * 70)
    print("SwinMLP Decomposition Verification")
    print("=" * 70)

    all_results = []

    # Level 0: Kernels
    print("\n=== Level 0: Kernels ===")
    level0_components = [
        ("Conv2d(3,96,k=4,s=4)", "level_0_kernel/conv2d_3x96_k4s4.py"),
        ("Flatten+Transpose", "level_0_kernel/flatten_transpose.py"),
        ("LayerNorm(96) PatchEmbed", "level_0_kernel/layer_norm_96.py"),
        ("Dropout(0.0)", "level_0_kernel/dropout_0.py"),
        ("LayerNorm(96) norm1", "level_0_kernel/layer_norm_96_block.py"),
        ("window_partition", "level_0_kernel/window_partition.py"),
        ("Spatial MLP Conv1d", "level_0_kernel/spatial_mlp_conv1d_96.py"),
        ("window_reverse", "level_0_kernel/window_reverse.py"),
        ("Residual Add", "level_0_kernel/residual_add.py"),
        ("LayerNorm(96) norm2", "level_0_kernel/layer_norm_96_2.py"),
        ("Linear(96,384)", "level_0_kernel/linear_96x384.py"),
        ("GELU", "level_0_kernel/gelu.py"),
        ("Linear(384,96)", "level_0_kernel/linear_384x96.py"),
        ("PatchMerging Downsample", "level_0_kernel/patch_merging_downsample.py"),
        ("LayerNorm(384)", "level_0_kernel/layer_norm_384.py"),
        ("Linear(384,192)", "level_0_kernel/linear_384x192.py"),
        ("LayerNorm(768)", "level_0_kernel/layer_norm_768.py"),
        ("AdaptiveAvgPool1d(1)", "level_0_kernel/adaptive_avg_pool1d.py"),
        ("Flatten(1)", "level_0_kernel/flatten.py"),
        ("Linear(768,1000)", "level_0_kernel/linear_768x1000.py"),
    ]
    for name, path in level0_components:
        all_results.append(test_component(name, path))

    # Level 1: Fusions
    print("\n=== Level 1: Fusions ===")
    level1_components = [
        ("Patch Proj + Norm", "level_1_fusion/patch_proj_norm.py"),
        ("SwinMLP Block Spatial", "level_1_fusion/swin_mlp_block_spatial.py"),
        ("SwinMLP Block FFN", "level_1_fusion/swin_mlp_block_ffn.py"),
        ("PatchMerging", "level_1_fusion/patch_merging.py"),
        ("Norm + AvgPool + Flatten", "level_1_fusion/norm_avgpool_flatten.py"),
    ]
    for name, path in level1_components:
        all_results.append(test_component(name, path))

    # Level 2: Layers
    print("\n=== Level 2: Layers ===")
    level2_components = [
        ("PatchEmbed", "level_2_layer/patch_embed.py"),
        ("BasicLayer Stage 0", "level_2_layer/basic_layer_0.py"),
        ("BasicLayer Stage 1", "level_2_layer/basic_layer_1.py"),
        ("BasicLayer Stage 2", "level_2_layer/basic_layer_2.py"),
        ("BasicLayer Stage 3", "level_2_layer/basic_layer_3.py"),
        ("Head", "level_2_layer/head.py"),
    ]
    for name, path in level2_components:
        all_results.append(test_component(name, path))

    # Level 3: Full Model
    print("\n=== Level 3: Full Model ===")
    all_results.append(test_component("SwinMLP (Full)", "level_3_model/swin_mlp.py"))

    # Data flow verification
    all_results.append(test_data_flow())

    # Summary
    passed = sum(all_results)
    total = len(all_results)
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {total - passed} test(s) failed")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
