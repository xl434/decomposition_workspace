"""
Composition Test for 38_LSTMBidirectional Decomposition
Verifies that all decomposition levels produce correct output shapes
and that the composition of components matches the full model behavior.
Source: data/kernelbench/level3/38_LSTMBidirectional.py
"""
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_level_0_kernels():
    """Test all Level 0 (kernel) components individually."""
    print("=" * 60)
    print("Testing Level 0 Kernels")
    print("=" * 60)
    all_passed = True

    # Test LSTM kernel
    print("\n--- lstm_128x256_6layers_bi ---")
    from level_0_kernel.lstm_128x256_6layers_bi import Model as LSTMModel
    model = LSTMModel(128, 256, 6, 0.0)
    model.eval()
    with torch.no_grad():
        x = torch.randn(10, 512, 128)
        h0 = torch.randn(12, 10, 256)
        c0 = torch.randn(12, 10, 256)
        out = model(x, h0, c0)
        assert out.shape == (10, 512, 512), f"LSTM output shape mismatch: {out.shape}"
        print(f"  Input: x={x.shape}, h0={h0.shape}, c0={c0.shape}")
        print(f"  Output: {out.shape}")
        print("  PASS")

    # Test slice kernel
    print("\n--- slice_last_timestep ---")
    from level_0_kernel.slice_last_timestep import Model as SliceModel
    model = SliceModel()
    model.eval()
    with torch.no_grad():
        x = torch.randn(10, 512, 512)
        out = model(x)
        assert out.shape == (10, 512), f"Slice output shape mismatch: {out.shape}"
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print("  PASS")

    # Test linear kernel
    print("\n--- linear_512x10 ---")
    from level_0_kernel.linear_512x10 import Model as LinearModel
    model = LinearModel(512, 10)
    model.eval()
    with torch.no_grad():
        x = torch.randn(10, 512)
        out = model(x)
        assert out.shape == (10, 10), f"Linear output shape mismatch: {out.shape}"
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print("  PASS")

    return all_passed


def test_level_1_fusions():
    """Test all Level 1 (fusion) components."""
    print("\n" + "=" * 60)
    print("Testing Level 1 Fusions")
    print("=" * 60)
    all_passed = True

    # Test BiLSTM fusion
    print("\n--- bilstm_fusion ---")
    from level_1_fusion.bilstm_fusion import Model as BiLSTMFusion
    model = BiLSTMFusion(128, 256, 6, 0.0)
    model.eval()
    with torch.no_grad():
        x = torch.randn(10, 512, 128)
        h0 = torch.randn(12, 10, 256)
        c0 = torch.randn(12, 10, 256)
        out = model(x, h0, c0)
        assert out.shape == (10, 512, 512), f"BiLSTM fusion shape mismatch: {out.shape}"
        print(f"  Input: x={x.shape}, h0={h0.shape}, c0={c0.shape}")
        print(f"  Output: {out.shape}")
        print("  PASS")

    # Test slice + linear fusion
    print("\n--- slice_linear ---")
    from level_1_fusion.slice_linear import Model as SliceLinearFusion
    model = SliceLinearFusion(512, 10)
    model.eval()
    with torch.no_grad():
        x = torch.randn(10, 512, 512)
        out = model(x)
        assert out.shape == (10, 10), f"Slice+Linear fusion shape mismatch: {out.shape}"
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print("  PASS")

    return all_passed


def test_level_2_layers():
    """Test all Level 2 (layer) components."""
    print("\n" + "=" * 60)
    print("Testing Level 2 Layers")
    print("=" * 60)
    all_passed = True

    # Test BiLSTM layer
    print("\n--- bilstm ---")
    from level_2_layer.bilstm import Model as BiLSTMLayer
    model = BiLSTMLayer(128, 256, 6, 0.0)
    model.eval()
    with torch.no_grad():
        x = torch.randn(10, 512, 128)
        h0 = torch.randn(12, 10, 256)
        c0 = torch.randn(12, 10, 256)
        out = model(x, h0, c0)
        assert out.shape == (10, 512, 512), f"BiLSTM layer shape mismatch: {out.shape}"
        print(f"  Input: x={x.shape}, h0={h0.shape}, c0={c0.shape}")
        print(f"  Output: {out.shape}")
        print("  PASS")

    # Test output layer
    print("\n--- output_layer ---")
    from level_2_layer.output_layer import Model as OutputLayer
    model = OutputLayer(512, 10)
    model.eval()
    with torch.no_grad():
        x = torch.randn(10, 512, 512)
        out = model(x)
        assert out.shape == (10, 10), f"Output layer shape mismatch: {out.shape}"
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        print("  PASS")

    return all_passed


def test_level_3_model():
    """Test the Level 3 (full model) component."""
    print("\n" + "=" * 60)
    print("Testing Level 3 Model")
    print("=" * 60)

    print("\n--- lstm_bidirectional (full model) ---")
    from level_3_model.lstm_bidirectional import Model as FullModel
    model = FullModel(128, 256, 6, 10, 0.0)
    model.eval()
    with torch.no_grad():
        x = torch.randn(10, 512, 128)
        h0 = torch.randn(12, 10, 256)
        c0 = torch.randn(12, 10, 256)
        out = model(x, h0, c0)
        assert out.shape == (10, 10), f"Full model shape mismatch: {out.shape}"
        print(f"  Input: x={x.shape}, h0={h0.shape}, c0={c0.shape}")
        print(f"  Output: {out.shape}")
        print("  PASS")

    return True


def test_composition():
    """Test that composing Level 0 kernels reproduces the full model data flow."""
    print("\n" + "=" * 60)
    print("Testing Composition (L0 kernels chained)")
    print("=" * 60)

    from level_0_kernel.lstm_128x256_6layers_bi import Model as LSTMKernel
    from level_0_kernel.slice_last_timestep import Model as SliceKernel
    from level_0_kernel.linear_512x10 import Model as LinearKernel

    # Build composed pipeline
    lstm = LSTMKernel(128, 256, 6, 0.0)
    slicer = SliceKernel()
    linear = LinearKernel(512, 10)

    lstm.eval()
    slicer.eval()
    linear.eval()

    with torch.no_grad():
        x = torch.randn(10, 512, 128)
        h0 = torch.randn(12, 10, 256)
        c0 = torch.randn(12, 10, 256)

        # Step 1: BiLSTM
        lstm_out = lstm(x, h0, c0)
        assert lstm_out.shape == (10, 512, 512), f"Step 1 shape: {lstm_out.shape}"
        print(f"  Step 1 (LSTM):  {x.shape} -> {lstm_out.shape}")

        # Step 2: Slice last timestep
        slice_out = slicer(lstm_out)
        assert slice_out.shape == (10, 512), f"Step 2 shape: {slice_out.shape}"
        print(f"  Step 2 (Slice): {lstm_out.shape} -> {slice_out.shape}")

        # Step 3: Linear
        final_out = linear(slice_out)
        assert final_out.shape == (10, 10), f"Step 3 shape: {final_out.shape}"
        print(f"  Step 3 (Linear): {slice_out.shape} -> {final_out.shape}")

        print(f"\n  Full pipeline: x={x.shape} -> {final_out.shape}")
        print("  PASS")

    return True


def test_numerical_consistency():
    """Test that the full model and composed kernels produce the same output
    when sharing the same weights."""
    print("\n" + "=" * 60)
    print("Testing Numerical Consistency (shared weights)")
    print("=" * 60)

    from level_3_model.lstm_bidirectional import Model as FullModel
    from level_0_kernel.lstm_128x256_6layers_bi import Model as LSTMKernel
    from level_0_kernel.slice_last_timestep import Model as SliceKernel
    from level_0_kernel.linear_512x10 import Model as LinearKernel

    # Create full model
    full_model = FullModel(128, 256, 6, 10, 0.0)
    full_model.eval()

    # Create individual kernels and copy weights
    lstm_kernel = LSTMKernel(128, 256, 6, 0.0)
    slice_kernel = SliceKernel()
    linear_kernel = LinearKernel(512, 10)

    # Copy LSTM weights
    lstm_kernel.lstm.load_state_dict(full_model.lstm.state_dict())
    # Copy Linear weights
    linear_kernel.fc.load_state_dict(full_model.fc.state_dict())

    lstm_kernel.eval()
    slice_kernel.eval()
    linear_kernel.eval()

    with torch.no_grad():
        x = torch.randn(10, 512, 128)
        h0 = torch.randn(12, 10, 256)
        c0 = torch.randn(12, 10, 256)

        # Full model forward
        full_out = full_model(x, h0, c0)

        # Composed forward
        lstm_out = lstm_kernel(x, h0, c0)
        slice_out = slice_kernel(lstm_out)
        composed_out = linear_kernel(slice_out)

        # Compare outputs
        max_diff = (full_out - composed_out).abs().max().item()
        print(f"  Full model output shape: {full_out.shape}")
        print(f"  Composed output shape:   {composed_out.shape}")
        print(f"  Max absolute difference: {max_diff:.2e}")

        assert max_diff < 1e-5, f"Numerical mismatch: max_diff={max_diff}"
        print("  PASS")

    return True


def main():
    results = {}
    results["Level 0 Kernels"] = test_level_0_kernels()
    results["Level 1 Fusions"] = test_level_1_fusions()
    results["Level 2 Layers"] = test_level_2_layers()
    results["Level 3 Model"] = test_level_3_model()
    results["Composition"] = test_composition()
    results["Numerical Consistency"] = test_numerical_consistency()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
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
    sys.exit(0 if main() else 1)
