"""
Composition Verification Test for 39_GRU Decomposition

Verifies that:
1. Each level (L0, L1, L2) produces correct output shapes
2. All levels produce numerically equivalent results when sharing weights
3. The decomposition hierarchy is consistent (L2 -> L1 -> L0)
4. The decomposed model matches the original model's behavior

Test structure:
  - test_level0_kernel: Verifies the GRU kernel produces correct output and h_n shapes
  - test_level1_fusion: Verifies the GRU fusion produces correct output shape
  - test_level2_layer: Verifies the complete model produces correct output shape
  - test_numerical_equivalence: Verifies all levels produce identical outputs with shared weights
  - test_original_model_match: Verifies L2 matches the original 39_GRU.py model exactly
"""
import torch
import torch.nn as nn
import sys
import os


def test_level0_kernel():
    """Test Level 0 (kernel): GRU kernel with 128x256, 6 layers."""
    print("=" * 60)
    print("Test Level 0 (Kernel): gru_128x256_6layers")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'level_0_kernel'))
    from gru_128x256_6layers import Model, get_inputs, get_init_inputs, get_expected_output_shape

    model = Model(*get_init_inputs())
    model.eval()

    with torch.no_grad():
        inputs = get_inputs()
        result = model(*inputs)

    # Kernel returns (output, h_n) tuple
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2 outputs, got {len(result)}"

    output, h_n = result
    assert tuple(output.shape) == (512, 10, 256), f"Output shape: {output.shape}"
    assert tuple(h_n.shape) == (6, 10, 256), f"h_n shape: {h_n.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isnan(h_n).any(), "NaN in h_n"
    assert not torch.isinf(output).any(), "Inf in output"
    assert not torch.isinf(h_n).any(), "Inf in h_n"

    expected = get_expected_output_shape()
    assert expected == [(512, 10, 256), (6, 10, 256)], f"Expected shapes: {expected}"

    print(f"  Input shapes:  x={inputs[0].shape}, h0={inputs[1].shape}")
    print(f"  Output shape:  {output.shape}")
    print(f"  h_n shape:     {h_n.shape}")
    print("  PASS")
    print()
    return model


def test_level1_fusion():
    """Test Level 1 (fusion): GRU fusion."""
    print("=" * 60)
    print("Test Level 1 (Fusion): gru_fusion")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'level_1_fusion'))
    from gru_fusion import Model, get_inputs, get_init_inputs, get_expected_output_shape

    model = Model(*get_init_inputs())
    model.eval()

    with torch.no_grad():
        inputs = get_inputs()
        output = model(*inputs)

    # Fusion returns only output (not h_n)
    assert not isinstance(output, tuple), f"Expected tensor, got tuple"
    assert tuple(output.shape) == (512, 10, 256), f"Output shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"

    expected = get_expected_output_shape()
    assert expected == [(512, 10, 256)], f"Expected shapes: {expected}"

    print(f"  Input shapes:  x={inputs[0].shape}, h0={inputs[1].shape}")
    print(f"  Output shape:  {output.shape}")
    print("  PASS")
    print()
    return model


def test_level2_layer():
    """Test Level 2 (layer): Complete GRU model."""
    print("=" * 60)
    print("Test Level 2 (Layer): gru_model")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'level_2_layer'))
    from gru_model import Model, get_inputs, get_init_inputs, get_expected_output_shape

    model = Model(*get_init_inputs())
    model.eval()

    with torch.no_grad():
        inputs = get_inputs()
        output = model(*inputs)

    # Layer returns only output (not h_n)
    assert not isinstance(output, tuple), f"Expected tensor, got tuple"
    assert tuple(output.shape) == (512, 10, 256), f"Output shape: {output.shape}"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"

    expected = get_expected_output_shape()
    assert expected == [(512, 10, 256)], f"Expected shapes: {expected}"

    print(f"  Input shapes:  x={inputs[0].shape}, h0={inputs[1].shape}")
    print(f"  Output shape:  {output.shape}")
    print("  PASS")
    print()
    return model


def test_numerical_equivalence():
    """
    Test that all levels produce identical outputs when sharing weights.

    Weight sharing: copy L2's GRU weights to L1 and L0, run same inputs,
    verify outputs match exactly (bitwise identical since same operations).
    """
    print("=" * 60)
    print("Test Numerical Equivalence Across Levels")
    print("=" * 60)

    # Import all levels
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'level_0_kernel'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'level_1_fusion'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'level_2_layer'))

    # Use fresh imports with unique names to avoid conflicts
    import importlib
    l0_mod = importlib.import_module('gru_128x256_6layers')
    l1_mod = importlib.import_module('gru_fusion')
    l2_mod = importlib.import_module('gru_model')

    # Create models
    l0_model = l0_mod.Model(*l0_mod.get_init_inputs())
    l1_model = l1_mod.Model(*l1_mod.get_init_inputs())
    l2_model = l2_mod.Model(*l2_mod.get_init_inputs())

    # Share weights: copy L2 weights to L1 and L0
    l1_model.gru.load_state_dict(l2_model.gru.state_dict())
    l0_model.gru.load_state_dict(l2_model.gru.state_dict())

    l0_model.eval()
    l1_model.eval()
    l2_model.eval()

    # Use fixed inputs for reproducibility
    torch.manual_seed(42)
    x = torch.randn(512, 10, 128)
    h0 = torch.randn(6, 10, 256)

    with torch.no_grad():
        l2_output = l2_model(x, h0)
        l1_output = l1_model(x, h0)
        l0_output, l0_h_n = l0_model(x, h0)

    # L2 vs L1: should be identical
    assert torch.equal(l2_output, l1_output), \
        f"L2 vs L1 mismatch! Max diff: {(l2_output - l1_output).abs().max().item()}"
    print("  L2 (layer) vs L1 (fusion): IDENTICAL")

    # L2 vs L0: should be identical (comparing output portion)
    assert torch.equal(l2_output, l0_output), \
        f"L2 vs L0 mismatch! Max diff: {(l2_output - l0_output).abs().max().item()}"
    print("  L2 (layer) vs L0 (kernel): IDENTICAL")

    # L1 vs L0: should be identical
    assert torch.equal(l1_output, l0_output), \
        f"L1 vs L0 mismatch! Max diff: {(l1_output - l0_output).abs().max().item()}"
    print("  L1 (fusion) vs L0 (kernel): IDENTICAL")

    print("  All levels numerically equivalent!")
    print("  PASS")
    print()


def test_original_model_match():
    """
    Test that the L2 decomposition matches the original 39_GRU model exactly.

    Reconstructs the original model inline and verifies numerical equivalence
    with the L2 decomposed model.
    """
    print("=" * 60)
    print("Test Original Model Match")
    print("=" * 60)

    # Original model (inline reconstruction)
    class OriginalModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
            super(OriginalModel, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first,
                              dropout=0, bidirectional=False)

        def forward(self, x, h0):
            output, h_n = self.gru(x, h0)
            return output

    # Decomposed L2 model
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'level_2_layer'))
    import importlib
    l2_mod = importlib.import_module('gru_model')

    original = OriginalModel(128, 256, 6)
    decomposed = l2_mod.Model(*l2_mod.get_init_inputs())

    # Share weights
    decomposed.gru.load_state_dict(original.gru.state_dict())

    original.eval()
    decomposed.eval()

    # Use fixed inputs
    torch.manual_seed(123)
    x = torch.randn(512, 10, 128)
    h0 = torch.randn(6, 10, 256)

    with torch.no_grad():
        orig_output = original(x, h0)
        decomp_output = decomposed(x, h0)

    assert torch.equal(orig_output, decomp_output), \
        f"Original vs decomposed mismatch! Max diff: {(orig_output - decomp_output).abs().max().item()}"

    print(f"  Original output shape:    {orig_output.shape}")
    print(f"  Decomposed output shape:  {decomp_output.shape}")
    print(f"  Outputs identical: True")
    print("  PASS")
    print()


def run_all_tests():
    """Run all composition verification tests."""
    print("*" * 60)
    print("  39_GRU Decomposition Verification")
    print("*" * 60)
    print()

    all_passed = True
    tests = [
        ("Level 0 Kernel", test_level0_kernel),
        ("Level 1 Fusion", test_level1_fusion),
        ("Level 2 Layer", test_level2_layer),
        ("Numerical Equivalence", test_numerical_equivalence),
        ("Original Model Match", test_original_model_match),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, "PASS"))
        except Exception as e:
            results.append((name, f"FAIL: {e}"))
            all_passed = False
            import traceback
            traceback.print_exc()
            print()

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")
    print()
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
