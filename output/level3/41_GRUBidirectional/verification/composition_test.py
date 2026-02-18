"""
Composition Test for 41_GRUBidirectional
Source: data/kernelbench/level3/41_GRUBidirectional.py

Verifies that all decomposition levels produce equivalent outputs and that
the decomposed components correctly reconstruct the original model behavior.

Tests:
1. Each level runs independently and produces correct output shapes
2. All levels produce numerically identical outputs when sharing weights
3. Original model matches the decomposed components
"""
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from level_0_kernel.gru_128x256_6layers_bi import Model as KernelModel
from level_0_kernel.gru_128x256_6layers_bi import get_inputs as kernel_get_inputs
from level_0_kernel.gru_128x256_6layers_bi import get_init_inputs as kernel_get_init_inputs

from level_1_fusion.bigru_fusion import Model as FusionModel
from level_1_fusion.bigru_fusion import get_inputs as fusion_get_inputs
from level_1_fusion.bigru_fusion import get_init_inputs as fusion_get_init_inputs

from level_2_layer.gru_bidirectional_model import Model as LayerModel
from level_2_layer.gru_bidirectional_model import get_inputs as layer_get_inputs
from level_2_layer.gru_bidirectional_model import get_init_inputs as layer_get_init_inputs


def test_individual_levels():
    """Test that each level runs independently and produces correct shapes."""
    print("=" * 60)
    print("Test 1: Individual level correctness")
    print("=" * 60)

    levels = [
        ("Level 0 (Kernel)", KernelModel, kernel_get_init_inputs, kernel_get_inputs),
        ("Level 1 (Fusion)", FusionModel, fusion_get_init_inputs, fusion_get_inputs),
        ("Level 2 (Layer)", LayerModel, layer_get_init_inputs, layer_get_inputs),
    ]

    expected_output_shape = (512, 10, 512)

    for name, ModelClass, get_init, get_inp in levels:
        model = ModelClass(*get_init())
        model.eval()
        with torch.no_grad():
            inputs = get_inp()
            output = model(*inputs)

            assert output.shape == expected_output_shape, \
                f"{name}: Expected shape {expected_output_shape}, got {output.shape}"
            assert not torch.isnan(output).any(), f"{name}: Output contains NaN"
            assert not torch.isinf(output).any(), f"{name}: Output contains Inf"

            print(f"  {name}: output shape = {output.shape} -- OK")

    print("  PASSED\n")
    return True


def test_cross_level_equivalence():
    """Test that all levels produce identical outputs when sharing weights."""
    print("=" * 60)
    print("Test 2: Cross-level weight-sharing equivalence")
    print("=" * 60)

    # Create reference model (Level 2 - top level)
    ref_model = LayerModel(*layer_get_init_inputs())
    ref_model.eval()

    # Create models at each level and copy weights from reference
    kernel_model = KernelModel(*kernel_get_init_inputs())
    kernel_model.eval()
    kernel_model.gru.load_state_dict(ref_model.gru.state_dict())

    fusion_model = FusionModel(*fusion_get_init_inputs())
    fusion_model.eval()
    fusion_model.gru.load_state_dict(ref_model.gru.state_dict())

    # Use identical inputs
    torch.manual_seed(42)
    inputs = layer_get_inputs()

    with torch.no_grad():
        ref_output = ref_model(*inputs)
        kernel_output = kernel_model(*inputs)
        fusion_output = fusion_model(*inputs)

    # Check exact equality (same weights, same inputs => same outputs)
    assert torch.equal(ref_output, kernel_output), \
        f"Kernel output differs from reference. Max diff: {(ref_output - kernel_output).abs().max().item()}"
    assert torch.equal(ref_output, fusion_output), \
        f"Fusion output differs from reference. Max diff: {(ref_output - fusion_output).abs().max().item()}"

    print(f"  Reference (Level 2) output shape: {ref_output.shape}")
    print(f"  Kernel (Level 0) matches reference: EXACT")
    print(f"  Fusion (Level 1) matches reference: EXACT")
    print("  PASSED\n")
    return True


def test_original_model_equivalence():
    """Test equivalence with the original model from the source."""
    print("=" * 60)
    print("Test 3: Original model equivalence")
    print("=" * 60)

    # Reconstruct original model
    class OriginalModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
            super(OriginalModel, self).__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
            # Note: original stores self.h0 but doesn't use it in forward
            self.h0 = torch.randn((num_layers * 2, 10, hidden_size))

        def forward(self, x, h0):
            output, h_n = self.gru(x, h0)
            return output

    original = OriginalModel(128, 256, 6)
    original.eval()

    decomposed = LayerModel(128, 256, 6)
    decomposed.eval()

    # Copy weights from original to decomposed
    decomposed.gru.load_state_dict(original.gru.state_dict())

    torch.manual_seed(123)
    x = torch.randn(512, 10, 128)
    h0 = torch.randn(12, 10, 256)

    with torch.no_grad():
        orig_output = original(x, h0)
        decomp_output = decomposed(x, h0)

    assert torch.equal(orig_output, decomp_output), \
        f"Decomposed output differs from original. Max diff: {(orig_output - decomp_output).abs().max().item()}"

    print(f"  Original output shape:     {orig_output.shape}")
    print(f"  Decomposed output shape:   {decomp_output.shape}")
    print(f"  Outputs are exactly equal: True")
    print("  PASSED\n")
    return True


def test_bidirectional_properties():
    """Test properties specific to bidirectional GRU."""
    print("=" * 60)
    print("Test 4: Bidirectional-specific properties")
    print("=" * 60)

    model = LayerModel(128, 256, 6)
    model.eval()

    # Verify GRU is bidirectional
    assert model.gru.bidirectional is True, "GRU should be bidirectional"
    print("  bidirectional=True: OK")

    # Verify number of parameters accounts for bidirectionality
    # Each layer has forward + backward parameters
    # Layer 0: input_size -> hidden_size (both directions)
    # Layers 1-5: hidden_size*2 -> hidden_size (both directions, input is concat of fwd+bwd)
    num_directions = 2
    expected_num_gate_params_layer0 = (
        num_directions * (3 * 256 * 128 + 3 * 256 + 3 * 256 * 256 + 3 * 256)
    )  # W_ih, b_ih, W_hh, b_hh for 3 gates, both directions
    # For layers 1-5, input size is hidden_size * num_directions = 512
    expected_num_gate_params_layer_n = (
        num_directions * (3 * 256 * 512 + 3 * 256 + 3 * 256 * 256 + 3 * 256)
    )

    total_expected = expected_num_gate_params_layer0 + 5 * expected_num_gate_params_layer_n
    actual_total = sum(p.numel() for p in model.gru.parameters())
    assert actual_total == total_expected, \
        f"Parameter count mismatch: expected {total_expected}, got {actual_total}"
    print(f"  Total GRU parameters: {actual_total} (matches expected {total_expected})")

    # Verify h0 dimensions: num_layers * num_directions = 12
    torch.manual_seed(0)
    h0 = torch.randn(12, 10, 256)
    x = torch.randn(512, 10, 128)
    with torch.no_grad():
        output = model(x, h0)

    # Output last dim should be hidden_size * 2 = 512
    assert output.shape[-1] == 512, f"Expected output last dim 512, got {output.shape[-1]}"
    print(f"  Output feature dim: {output.shape[-1]} (hidden_size*2=512): OK")

    # Verify forward and backward halves can differ
    fwd_output = output[:, :, :256]   # First 256 features (forward direction)
    bwd_output = output[:, :, 256:]   # Last 256 features (backward direction)
    assert not torch.equal(fwd_output, bwd_output), \
        "Forward and backward outputs should differ (different random weights and directions)"
    print("  Forward/backward outputs differ: OK")

    print("  PASSED\n")
    return True


def test_determinism():
    """Test that the model produces deterministic outputs."""
    print("=" * 60)
    print("Test 5: Determinism")
    print("=" * 60)

    model = LayerModel(128, 256, 6)
    model.eval()

    torch.manual_seed(99)
    inputs1 = layer_get_inputs()
    torch.manual_seed(99)
    inputs2 = layer_get_inputs()

    with torch.no_grad():
        out1 = model(*inputs1)
        out2 = model(*inputs2)

    assert torch.equal(out1, out2), "Same inputs should produce same outputs"
    print("  Same inputs produce identical outputs: OK")
    print("  PASSED\n")
    return True


def run_all_tests():
    """Run all composition tests."""
    print("Composition Tests for 41_GRUBidirectional")
    print("=" * 60)
    print()

    results = []
    results.append(("Individual levels", test_individual_levels()))
    results.append(("Cross-level equivalence", test_cross_level_equivalence()))
    results.append(("Original model equivalence", test_original_model_equivalence()))
    results.append(("Bidirectional properties", test_bidirectional_properties()))
    results.append(("Determinism", test_determinism()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
