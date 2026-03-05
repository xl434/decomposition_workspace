"""
Level 3 Model: SqueezeNet (Model 18 from KernelBench Level 3)
Full SqueezeNet architecture with 8 Fire Modules.

Composed of L2 layers:
  - initial_block: Conv(3,96,7,s=2) + ReLU + MaxPool
  - fire_module_1: FireModule(96, 16, 64, 64)
  - fire_module_2: FireModule(128, 16, 64, 64)
  - fire_module_3: FireModule(128, 32, 128, 128)
  - MaxPool2d(3,2,ceil)
  - fire_module_4: FireModule(256, 32, 128, 128)
  - fire_module_5: FireModule(256, 48, 192, 192)
  - fire_module_6: FireModule(384, 48, 192, 192)
  - fire_module_7: FireModule(384, 64, 256, 256)
  - MaxPool2d(3,2,ceil)
  - fire_module_8: FireModule(512, 64, 256, 256)
  - classifier_block: Dropout + Conv(512,10,1) + ReLU + AdaptiveAvgPool(1,1) + Flatten

Input: [2, 3, 32, 32] -> Output: [2, 10]

Shape trace:
  Conv2d(3,96,7,s=2):       [2,3,32,32]   -> [2,96,13,13]
  ReLU:                      [2,96,13,13]  -> [2,96,13,13]
  MaxPool2d(3,2,ceil):       [2,96,13,13]  -> [2,96,7,7]
  Fire1(96,16,64,64):        [2,96,7,7]    -> [2,128,7,7]
  Fire2(128,16,64,64):       [2,128,7,7]   -> [2,128,7,7]
  Fire3(128,32,128,128):     [2,128,7,7]   -> [2,256,7,7]
  MaxPool2d(3,2,ceil):       [2,256,7,7]   -> [2,256,4,4]
  Fire4(256,32,128,128):     [2,256,4,4]   -> [2,256,4,4]
  Fire5(256,48,192,192):     [2,256,4,4]   -> [2,384,4,4]
  Fire6(384,48,192,192):     [2,384,4,4]   -> [2,384,4,4]
  Fire7(384,64,256,256):     [2,384,4,4]   -> [2,512,4,4]
  MaxPool2d(3,2,ceil):       [2,512,4,4]   -> [2,512,2,2]
  Fire8(512,64,256,256):     [2,512,2,2]   -> [2,512,2,2]
  Dropout(0.0):              [2,512,2,2]   -> [2,512,2,2]
  Conv2d(512,10,1):          [2,512,2,2]   -> [2,10,2,2]
  ReLU:                      [2,10,2,2]    -> [2,10,2,2]
  AdaptiveAvgPool2d(1,1):    [2,10,2,2]    -> [2,10,1,1]
  Flatten:                   [2,10,1,1]    -> [2,10]
"""

import torch
import torch.nn as nn


class FireModule(nn.Module):
    """Fire Module: squeeze layer feeding parallel expand layers."""

    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class Model(nn.Module):
    """Full SqueezeNet model with 8 Fire Modules."""

    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),       # Fire1
            FireModule(128, 16, 64, 64),      # Fire2
            FireModule(128, 32, 128, 128),    # Fire3
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),    # Fire4
            FireModule(256, 48, 192, 192),    # Fire5
            FireModule(384, 48, 192, 192),    # Fire6
            FireModule(384, 64, 256, 256),    # Fire7
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),    # Fire8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(2, 3, 32, 32)]


def get_init_inputs():
    """Return list of arguments for Model.__init__."""
    return []  # Use default: num_classes=10


def get_expected_output_shape():
    """Return expected output shape."""
    return (2, 10)


def run_tests():
    """Verify full SqueezeNet produces correct output shape."""
    model = Model(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    expected_shape = get_expected_output_shape()
    assert output.shape == torch.Size(expected_shape), \
        f"Shape mismatch: got {output.shape}, expected {expected_shape}"
    print(f"PASSED: squeezenet output shape {output.shape}")

    # Verify intermediate shapes by running step-by-step
    x = inputs[0]
    with torch.no_grad():
        # Initial block
        x = model.features[0](x)  # Conv2d
        assert x.shape == (2, 96, 13, 13), f"After initial conv: {x.shape}"
        x = model.features[1](x)  # ReLU
        assert x.shape == (2, 96, 13, 13), f"After initial relu: {x.shape}"
        x = model.features[2](x)  # MaxPool
        assert x.shape == (2, 96, 7, 7), f"After initial pool: {x.shape}"

        # Fire modules 1-3
        x = model.features[3](x)  # Fire1
        assert x.shape == (2, 128, 7, 7), f"After fire1: {x.shape}"
        x = model.features[4](x)  # Fire2
        assert x.shape == (2, 128, 7, 7), f"After fire2: {x.shape}"
        x = model.features[5](x)  # Fire3
        assert x.shape == (2, 256, 7, 7), f"After fire3: {x.shape}"

        # Pool
        x = model.features[6](x)  # MaxPool
        assert x.shape == (2, 256, 4, 4), f"After pool2: {x.shape}"

        # Fire modules 4-7
        x = model.features[7](x)  # Fire4
        assert x.shape == (2, 256, 4, 4), f"After fire4: {x.shape}"
        x = model.features[8](x)  # Fire5
        assert x.shape == (2, 384, 4, 4), f"After fire5: {x.shape}"
        x = model.features[9](x)  # Fire6
        assert x.shape == (2, 384, 4, 4), f"After fire6: {x.shape}"
        x = model.features[10](x)  # Fire7
        assert x.shape == (2, 512, 4, 4), f"After fire7: {x.shape}"

        # Pool
        x = model.features[11](x)  # MaxPool
        assert x.shape == (2, 512, 2, 2), f"After pool3: {x.shape}"

        # Fire8
        x = model.features[12](x)  # Fire8
        assert x.shape == (2, 512, 2, 2), f"After fire8: {x.shape}"

        # Classifier
        x = model.classifier(x)
        assert x.shape == (2, 10, 1, 1), f"After classifier: {x.shape}"
        x = torch.flatten(x, 1)
        assert x.shape == (2, 10), f"After flatten: {x.shape}"

    print("PASSED: All intermediate shapes verified")
    return True


if __name__ == "__main__":
    run_tests()
