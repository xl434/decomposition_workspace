"""
Level 2 Layer: LeNet-5 (Full Model)
Operations: Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d
            -> Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear
Input: [2, 1, 32, 32] -> Output: [2, 10]

Full LeNet-5 model (4_LeNet5) from KernelBench Level 3.
Classic convolutional neural network for digit classification.

Composed of Level 1 fusions:
  1. conv_relu_pool_1:  Conv2d(1,6,5) + ReLU + MaxPool2d(2,2)   [2,1,32,32] -> [2,6,14,14]
  2. conv_relu_pool_2:  Conv2d(6,16,5) + ReLU + MaxPool2d(2,2)  [2,6,14,14] -> [2,16,5,5]
  3. flatten_fc_relu_1: Flatten + Linear(400,120) + ReLU         [2,16,5,5]  -> [2,120]
  4. fc_relu_2:         Linear(120,84) + ReLU                    [2,120]     -> [2,84]
  5. fc_output:         Linear(84,10)                            [2,84]      -> [2,10]

Total operations: 12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """LeNet-5: Classic CNN for image classification."""

    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 1, 32, 32]
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Block 1: conv_relu_pool_1
        x = F.relu(self.conv1(x))       # [B,1,32,32] -> [B,6,28,28]
        x = F.max_pool2d(x, 2, 2)       # [B,6,28,28] -> [B,6,14,14]

        # Block 2: conv_relu_pool_2
        x = F.relu(self.conv2(x))       # [B,6,14,14] -> [B,16,10,10]
        x = F.max_pool2d(x, 2, 2)       # [B,16,10,10] -> [B,16,5,5]

        # Block 3: flatten_fc_relu_1
        x = x.view(-1, 16 * 5 * 5)     # [B,16,5,5] -> [B,400]
        x = F.relu(self.fc1(x))         # [B,400] -> [B,120]

        # Block 4: fc_relu_2
        x = F.relu(self.fc2(x))         # [B,120] -> [B,84]

        # Block 5: fc_output
        x = self.fc3(x)                 # [B,84] -> [B,num_classes]

        return x


def get_inputs():
    """Return list of input tensors for the full model."""
    batch_size = 2
    return [torch.randn(batch_size, 1, 32, 32)]


def get_init_inputs():
    """Return list of constructor arguments."""
    return [10]  # num_classes


def get_expected_output_shape():
    """Return the expected output shape."""
    return (2, 10)


def run_tests():
    """Validate the full model produces correct output shapes and runs without error."""
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)

    expected_shape = get_expected_output_shape()
    assert output.shape == expected_shape, (
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    )

    # Test that gradients flow end-to-end
    x = inputs[0].clone().requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back to input"
    assert x.grad.shape == x.shape, "Gradient shape should match input shape"

    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    # conv1: 1*6*5*5+6=156, conv2: 6*16*5*5+16=2416
    # fc1: 400*120+120=48120, fc2: 120*84+84=10164, fc3: 84*10+10=850
    expected_params = 156 + 2416 + 48120 + 10164 + 850
    assert total_params == expected_params, (
        f"Parameter count mismatch: got {total_params}, expected {expected_params}"
    )

    # Verify intermediate shapes by hooking
    intermediate_shapes = []
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            intermediate_shapes.append((name, output.shape))
        return hook_fn

    hooks.append(model.conv1.register_forward_hook(make_hook("conv1")))
    hooks.append(model.conv2.register_forward_hook(make_hook("conv2")))
    hooks.append(model.fc1.register_forward_hook(make_hook("fc1")))
    hooks.append(model.fc2.register_forward_hook(make_hook("fc2")))
    hooks.append(model.fc3.register_forward_hook(make_hook("fc3")))

    model.eval()
    with torch.no_grad():
        _ = model(torch.randn(2, 1, 32, 32))

    for h in hooks:
        h.remove()

    expected_intermediates = {
        "conv1": (2, 6, 28, 28),
        "conv2": (2, 16, 10, 10),
        "fc1": (2, 120),
        "fc2": (2, 84),
        "fc3": (2, 10),
    }

    for name, shape in intermediate_shapes:
        assert tuple(shape) == expected_intermediates[name], (
            f"Intermediate shape mismatch at {name}: got {tuple(shape)}, "
            f"expected {expected_intermediates[name]}"
        )

    print("lenet5 (Level 2): All tests passed!")
    print(f"  Input shape:  {inputs[0].shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Total parameters: {total_params}")
    return True


if __name__ == "__main__":
    run_tests()
