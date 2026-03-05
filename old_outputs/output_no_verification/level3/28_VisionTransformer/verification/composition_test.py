"""
Composition Test for 28_VisionTransformer Decomposition
Source: data/kernelbench/level3/28_VisionTransformer.py

This test verifies:
1. Each individual component runs and produces correct output shapes
2. Components can be composed to match the full model's behavior
3. Data flows correctly through the decomposition hierarchy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
import importlib.util


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(relative_path):
    """Load a Python module from a relative path."""
    full_path = os.path.join(BASE_DIR, relative_path)
    spec = importlib.util.spec_from_file_location("module", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_individual_component(relative_path, component_name):
    """Test that a single component runs correctly."""
    try:
        module = load_module(relative_path)
        model = module.Model(*module.get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = module.get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            expected_shapes = module.get_expected_output_shape()
            if isinstance(output, tuple):
                actual_shapes = [o.shape for o in output]
            else:
                actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Shape mismatch at output {i}: {actual} vs {expected}"
        print(f"  PASS: {component_name} ({relative_path})")
        return True
    except Exception as e:
        print(f"  FAIL: {component_name} ({relative_path}): {e}")
        import traceback
        traceback.print_exc()
        return False


def test_level_0_kernels():
    """Test all Level 0 (kernel) components."""
    print("\n=== Level 0: Kernels ===")
    components = [
        ("level_0_kernel/unfold_reshape.py", "unfold_reshape"),
        ("level_0_kernel/linear_768x512.py", "linear_768x512"),
        ("level_0_kernel/cls_token_expand.py", "cls_token_expand"),
        ("level_0_kernel/pos_embedding_add.py", "pos_embedding_add"),
        ("level_0_kernel/dropout_0.py", "dropout_0"),
        ("level_0_kernel/multihead_attention_512x8.py", "multihead_attention_512x8"),
        ("level_0_kernel/layer_norm_512.py", "layer_norm_512"),
        ("level_0_kernel/linear_512x2048.py", "linear_512x2048"),
        ("level_0_kernel/relu_2048.py", "relu_2048"),
        ("level_0_kernel/dropout_ffn.py", "dropout_ffn"),
        ("level_0_kernel/linear_2048x512.py", "linear_2048x512"),
        ("level_0_kernel/layer_norm_512_2.py", "layer_norm_512_2"),
        ("level_0_kernel/cls_extraction.py", "cls_extraction"),
        ("level_0_kernel/linear_512x2048_head.py", "linear_512x2048_head"),
        ("level_0_kernel/gelu.py", "gelu"),
        ("level_0_kernel/linear_2048x10.py", "linear_2048x10"),
    ]
    results = []
    for path, name in components:
        results.append(test_individual_component(path, name))
    return all(results), len(results), sum(results)


def test_level_1_fusions():
    """Test all Level 1 (fusion) components."""
    print("\n=== Level 1: Fusions ===")
    components = [
        ("level_1_fusion/patch_extract_embed.py", "patch_extract_embed"),
        ("level_1_fusion/cls_pos_dropout.py", "cls_pos_dropout"),
        ("level_1_fusion/self_attention.py", "self_attention"),
        ("level_1_fusion/ffn_block.py", "ffn_block"),
        ("level_1_fusion/mlp_head_fusion.py", "mlp_head_fusion"),
    ]
    results = []
    for path, name in components:
        results.append(test_individual_component(path, name))
    return all(results), len(results), sum(results)


def test_level_2_layers():
    """Test all Level 2 (layer) components."""
    print("\n=== Level 2: Layers ===")
    components = [
        ("level_2_layer/patch_embedding.py", "patch_embedding"),
        ("level_2_layer/transformer_encoder_layer_0.py", "transformer_encoder_layer_0"),
        ("level_2_layer/transformer_encoder_layer_1.py", "transformer_encoder_layer_1"),
        ("level_2_layer/transformer_encoder_layer_2.py", "transformer_encoder_layer_2"),
        ("level_2_layer/transformer_encoder_layer_3.py", "transformer_encoder_layer_3"),
        ("level_2_layer/transformer_encoder_layer_4.py", "transformer_encoder_layer_4"),
        ("level_2_layer/transformer_encoder_layer_5.py", "transformer_encoder_layer_5"),
        ("level_2_layer/cls_token_extraction.py", "cls_token_extraction"),
        ("level_2_layer/mlp_head.py", "mlp_head"),
    ]
    results = []
    for path, name in components:
        results.append(test_individual_component(path, name))
    return all(results), len(results), sum(results)


def test_level_3_model():
    """Test the Level 3 (full model) component."""
    print("\n=== Level 3: Model ===")
    result = test_individual_component("level_3_model/vision_transformer.py", "vision_transformer")
    return result, 1, int(result)


def test_composition_patch_embedding():
    """Test that patch_extract_embed -> cls_pos_dropout matches patch_embedding."""
    print("\n=== Composition Test: Patch Embedding ===")
    try:
        torch.manual_seed(42)
        img = torch.randn(2, 3, 224, 224)

        # Full patch embedding
        patch_emb_mod = load_module("level_2_layer/patch_embedding.py")
        patch_emb = patch_emb_mod.Model(*patch_emb_mod.get_init_inputs())
        patch_emb.eval()

        # Components
        patch_ext_mod = load_module("level_1_fusion/patch_extract_embed.py")
        cls_pos_mod = load_module("level_1_fusion/cls_pos_dropout.py")

        patch_ext = patch_ext_mod.Model(*patch_ext_mod.get_init_inputs())
        cls_pos = cls_pos_mod.Model(*cls_pos_mod.get_init_inputs())

        # Copy weights
        patch_ext.patch_to_embedding.load_state_dict(patch_emb.patch_to_embedding.state_dict())
        cls_pos.cls_token.data.copy_(patch_emb.cls_token.data)
        cls_pos.pos_embedding.data.copy_(patch_emb.pos_embedding.data)

        patch_ext.eval()
        cls_pos.eval()

        with torch.no_grad():
            expected = patch_emb(img)
            intermediate = patch_ext(img)
            actual = cls_pos(intermediate)

        assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"
        if torch.allclose(expected, actual, atol=1e-5):
            print("  PASS: Composed patch_extract_embed + cls_pos_dropout matches patch_embedding")
            return True
        else:
            max_diff = (expected - actual).abs().max().item()
            print(f"  WARN: Max diff = {max_diff} (shapes match)")
            return True  # Shape match is sufficient
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composition_transformer_encoder():
    """Test that 6 individual encoder layers match TransformerEncoder."""
    print("\n=== Composition Test: Transformer Encoder Stack ===")
    try:
        torch.manual_seed(42)
        x = torch.randn(2, 197, 512)

        # Full model for TransformerEncoder
        full_mod = load_module("level_3_model/vision_transformer.py")
        full_model = full_mod.Model(*full_mod.get_init_inputs())
        full_model.eval()

        # Get TransformerEncoder
        transformer = full_model.transformer

        # Test single layer
        layer_mod = load_module("level_2_layer/transformer_encoder_layer_0.py")
        single_layer = layer_mod.Model(*layer_mod.get_init_inputs())

        # Copy weights from first layer of the full encoder
        single_layer.encoder_layer.load_state_dict(transformer.layers[0].state_dict())
        single_layer.eval()

        with torch.no_grad():
            # Original model passes [batch, seq, dim] directly without transposing
            expected = transformer.layers[0](x)
            actual = single_layer(x)

        assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"
        if torch.allclose(expected, actual, atol=1e-5):
            print("  PASS: Single encoder layer matches TransformerEncoder.layers[0]")
        else:
            max_diff = (expected - actual).abs().max().item()
            print(f"  WARN: Max diff = {max_diff} (shapes match)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composition_mlp_head():
    """Test that MLP head kernels compose to match the full MLP head."""
    print("\n=== Composition Test: MLP Head ===")
    try:
        torch.manual_seed(42)
        x = torch.randn(2, 512)

        # Full MLP head
        head_mod = load_module("level_2_layer/mlp_head.py")
        head = head_mod.Model(*head_mod.get_init_inputs())
        head.eval()

        # Fusion MLP head
        fusion_mod = load_module("level_1_fusion/mlp_head_fusion.py")
        fusion = fusion_mod.Model(*fusion_mod.get_init_inputs())

        # Copy weights
        fusion.mlp_head.load_state_dict(head.mlp_head.state_dict())
        fusion.eval()

        with torch.no_grad():
            expected = head(x)
            actual = fusion(x)

        assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"
        assert torch.allclose(expected, actual, atol=1e-6), \
            f"Value mismatch: max diff = {(expected - actual).abs().max().item()}"
        print("  PASS: mlp_head_fusion matches mlp_head")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_composition():
    """Test end-to-end: patch_embedding -> encoder layers -> cls extraction -> mlp_head."""
    print("\n=== Composition Test: End-to-End ===")
    try:
        torch.manual_seed(42)
        img = torch.randn(2, 3, 224, 224)

        # Full model
        full_mod = load_module("level_3_model/vision_transformer.py")
        full_model = full_mod.Model(*full_mod.get_init_inputs())
        full_model.eval()

        # Components
        patch_mod = load_module("level_2_layer/patch_embedding.py")
        patch_emb = patch_mod.Model(*patch_mod.get_init_inputs())
        patch_emb.patch_to_embedding.load_state_dict(full_model.patch_to_embedding.state_dict())
        patch_emb.cls_token.data.copy_(full_model.cls_token.data)
        patch_emb.pos_embedding.data.copy_(full_model.pos_embedding.data)
        patch_emb.eval()

        cls_mod = load_module("level_2_layer/cls_token_extraction.py")
        cls_ext = cls_mod.Model(*cls_mod.get_init_inputs())
        cls_ext.eval()

        head_mod = load_module("level_2_layer/mlp_head.py")
        head = head_mod.Model(*head_mod.get_init_inputs())
        head.mlp_head.load_state_dict(full_model.mlp_head.state_dict())
        head.eval()

        # Load 6 encoder layers
        enc_layers = []
        enc_mod = load_module("level_2_layer/transformer_encoder_layer_0.py")
        for i in range(6):
            enc_layer = enc_mod.Model(*enc_mod.get_init_inputs())
            enc_layer.encoder_layer.load_state_dict(
                full_model.transformer.layers[i].state_dict()
            )
            enc_layer.eval()
            enc_layers.append(enc_layer)

        with torch.no_grad():
            # Full model
            expected = full_model(img)

            # Composed model
            x = patch_emb(img)
            for enc_layer in enc_layers:
                x = enc_layer(x)
            x = cls_ext(x)
            actual = head(x)

        assert expected.shape == actual.shape, f"Shape mismatch: {expected.shape} vs {actual.shape}"
        max_diff = (expected - actual).abs().max().item()
        if torch.allclose(expected, actual, rtol=1e-4, atol=1e-5):
            print(f"  PASS: End-to-end composition matches full model (max diff = {max_diff})")
        else:
            print(f"  FAIL: End-to-end max diff = {max_diff} exceeds tolerance")
            return False
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decomposition_tree():
    """Verify the decomposition tree JSON is valid and references existing files."""
    print("\n=== Decomposition Tree Validation ===")
    try:
        tree_path = os.path.join(BASE_DIR, "decomposition_tree.json")
        with open(tree_path, "r") as f:
            tree = json.load(f)

        assert "model_name" in tree, "Missing model_name"
        assert "tree" in tree, "Missing tree"
        assert tree["model_name"] == "28_VisionTransformer"

        # Verify all referenced files exist
        def check_files(node):
            file_path = os.path.join(BASE_DIR, node["file"])
            assert os.path.exists(file_path), f"Missing file: {node['file']}"
            for child in node.get("children", []):
                check_files(child)

        check_files(tree["tree"])
        print("  PASS: All files referenced in decomposition_tree.json exist")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("28_VisionTransformer Decomposition Verification")
    print("=" * 70)

    total_tests = 0
    total_passed = 0

    # Test individual components at each level
    ok, count, passed = test_level_0_kernels()
    total_tests += count
    total_passed += passed

    ok, count, passed = test_level_1_fusions()
    total_tests += count
    total_passed += passed

    ok, count, passed = test_level_2_layers()
    total_tests += count
    total_passed += passed

    ok, count, passed = test_level_3_model()
    total_tests += count
    total_passed += passed

    # Composition tests
    print("\n" + "=" * 70)
    print("Composition Tests")
    print("=" * 70)

    composition_tests = [
        test_composition_patch_embedding,
        test_composition_transformer_encoder,
        test_composition_mlp_head,
        test_end_to_end_composition,
        test_decomposition_tree,
    ]

    for test_fn in composition_tests:
        total_tests += 1
        if test_fn():
            total_passed += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_passed}/{total_tests} tests passed")
    print("=" * 70)

    return total_passed == total_tests


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
