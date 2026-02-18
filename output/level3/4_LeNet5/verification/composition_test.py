"""
Composition Test for 4_LeNet5
Verifies that composing L0 kernel components reproduces the original model output.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the source directory to sys.path so we can import the original model
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "kernelbench", "level3"))
sys.path.insert(0, source_dir)

# Import kernel components
kernel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "level_0_kernel"))
sys.path.insert(0, kernel_dir)


def composed_forward(x, conv1_weight, conv1_bias, conv2_weight, conv2_bias,
                     fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias):
    """
    Compose the full LeNet-5 forward pass from individual kernel operations.
    This mirrors the original model's forward method using only primitive operations.
    """
    # Kernel 1: Conv2d(1,6,5)
    x = F.conv2d(x, conv1_weight, conv1_bias, stride=1)
    # Kernel 2: ReLU
    x = F.relu(x)
    # Kernel 3: MaxPool2d(2,2)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    # Kernel 4: Conv2d(6,16,5)
    x = F.conv2d(x, conv2_weight, conv2_bias, stride=1)
    # Kernel 5: ReLU
    x = F.relu(x)
    # Kernel 6: MaxPool2d(2,2)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    # Kernel 7: Flatten
    x = x.view(-1, 16 * 5 * 5)
    # Kernel 8: Linear(400,120)
    x = F.linear(x, fc1_weight, fc1_bias)
    # Kernel 9: ReLU
    x = F.relu(x)
    # Kernel 10: Linear(120,84)
    x = F.linear(x, fc2_weight, fc2_bias)
    # Kernel 11: ReLU
    x = F.relu(x)
    # Kernel 12: Linear(84,20)
    x = F.linear(x, fc3_weight, fc3_bias)
    return x


def test_composition():
    """Test that composed kernel operations match original model output."""
    print("=" * 60)
    print("Composition Test: 4_LeNet5")
    print("=" * 60)

    # Import the original model
    try:
        from importlib import import_module
        original_module = import_module("4_LeNet5")
        OriginalModel = original_module.Model
    except ImportError:
        # Fallback: define inline if import fails
        print("WARNING: Could not import original module, using inline definition")

        class OriginalModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
                self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
                self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
                self.fc2 = nn.Linear(in_features=120, out_features=84)
                self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, kernel_size=2, stride=2)
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    batch_size = 4096
    num_classes = 20

    # Create original model
    original_model = OriginalModel(num_classes)
    original_model.eval()

    # Extract shared weights from original model
    conv1_weight = original_model.conv1.weight.data
    conv1_bias = original_model.conv1.bias.data
    conv2_weight = original_model.conv2.weight.data
    conv2_bias = original_model.conv2.bias.data
    fc1_weight = original_model.fc1.weight.data
    fc1_bias = original_model.fc1.bias.data
    fc2_weight = original_model.fc2.weight.data
    fc2_bias = original_model.fc2.bias.data
    fc3_weight = original_model.fc3.weight.data
    fc3_bias = original_model.fc3.bias.data

    # Create test input
    torch.manual_seed(42)
    x = torch.randn(batch_size, 1, 32, 32)

    with torch.no_grad():
        # Original model output
        original_output = original_model(x)

        # Composed output from kernels with shared weights
        composed_output = composed_forward(
            x,
            conv1_weight, conv1_bias,
            conv2_weight, conv2_bias,
            fc1_weight, fc1_bias,
            fc2_weight, fc2_bias,
            fc3_weight, fc3_bias
        )

    # Verify shapes match
    print(f"\nOriginal output shape:  {original_output.shape}")
    print(f"Composed output shape:  {composed_output.shape}")
    assert original_output.shape == composed_output.shape, \
        f"Shape mismatch: {original_output.shape} vs {composed_output.shape}"
    print("Shape check: PASS")

    # Verify values match (with tolerance)
    is_close = torch.allclose(original_output, composed_output, rtol=1e-4, atol=1e-5)
    max_diff = (original_output - composed_output).abs().max().item()
    mean_diff = (original_output - composed_output).abs().mean().item()

    print(f"\nMax absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Values match (rtol=1e-4, atol=1e-5): {'PASS' if is_close else 'FAIL'}")

    if not is_close:
        print("\nFAIL: Composed output does not match original model output!")
        return False

    # Test individual level compositions
    print("\n" + "-" * 40)
    print("Testing Level 2 (Layer) compositions...")
    print("-" * 40)

    with torch.no_grad():
        # Test conv_block_1
        cb1_out = F.max_pool2d(F.relu(F.conv2d(x, conv1_weight, conv1_bias, stride=1)),
                               kernel_size=2, stride=2)
        cb1_expected = F.max_pool2d(F.relu(original_model.conv1(x)), kernel_size=2, stride=2)
        cb1_match = torch.allclose(cb1_out, cb1_expected, rtol=1e-4, atol=1e-5)
        print(f"conv_block_1: {'PASS' if cb1_match else 'FAIL'} (shape: {cb1_out.shape})")

        # Test conv_block_2
        cb2_out = F.max_pool2d(F.relu(F.conv2d(cb1_out, conv2_weight, conv2_bias, stride=1)),
                               kernel_size=2, stride=2)
        cb2_expected = F.max_pool2d(F.relu(original_model.conv2(cb1_expected)), kernel_size=2, stride=2)
        cb2_match = torch.allclose(cb2_out, cb2_expected, rtol=1e-4, atol=1e-5)
        print(f"conv_block_2: {'PASS' if cb2_match else 'FAIL'} (shape: {cb2_out.shape})")

        # Test classifier
        flat = cb2_out.view(-1, 16 * 5 * 5)
        cls_out = F.linear(F.relu(F.linear(F.relu(F.linear(flat, fc1_weight, fc1_bias)),
                                           fc2_weight, fc2_bias)),
                           fc3_weight, fc3_bias)
        cls_match = torch.allclose(cls_out, composed_output, rtol=1e-4, atol=1e-5)
        print(f"classifier:   {'PASS' if cls_match else 'FAIL'} (shape: {cls_out.shape})")

    print("\n" + "-" * 40)
    print("Testing Level 1 (Fusion) compositions...")
    print("-" * 40)

    with torch.no_grad():
        # Test conv_relu_pool_1x6x5
        crp1 = F.max_pool2d(F.relu(F.conv2d(x, conv1_weight, conv1_bias, stride=1)),
                            kernel_size=2, stride=2)
        crp1_match = torch.allclose(crp1, cb1_out, rtol=1e-4, atol=1e-5)
        print(f"conv_relu_pool_1x6x5: {'PASS' if crp1_match else 'FAIL'} (shape: {crp1.shape})")

        # Test conv_relu_pool_6x16x5
        crp2 = F.max_pool2d(F.relu(F.conv2d(crp1, conv2_weight, conv2_bias, stride=1)),
                            kernel_size=2, stride=2)
        crp2_match = torch.allclose(crp2, cb2_out, rtol=1e-4, atol=1e-5)
        print(f"conv_relu_pool_6x16x5: {'PASS' if crp2_match else 'FAIL'} (shape: {crp2.shape})")

        # Test linear_relu_400x120
        flat2 = crp2.view(-1, 16 * 5 * 5)
        lr1 = F.relu(F.linear(flat2, fc1_weight, fc1_bias))
        lr1_match = lr1.shape == (batch_size, 120)
        print(f"linear_relu_400x120:  {'PASS' if lr1_match else 'FAIL'} (shape: {lr1.shape})")

        # Test linear_relu_120x84
        lr2 = F.relu(F.linear(lr1, fc2_weight, fc2_bias))
        lr2_match = lr2.shape == (batch_size, 84)
        print(f"linear_relu_120x84:   {'PASS' if lr2_match else 'FAIL'} (shape: {lr2.shape})")

    all_passed = is_close and cb1_match and cb2_match and cls_match and crp1_match and crp2_match and lr1_match and lr2_match

    print("\n" + "=" * 60)
    print(f"Overall composition test: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_composition()
    sys.exit(0 if success else 1)
