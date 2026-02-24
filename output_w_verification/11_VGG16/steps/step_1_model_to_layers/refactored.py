"""
Refactored VGG16 Model: Model (L3) -> Layers (L2)

The forward() only calls child modules + data plumbing (flatten).
All compute lives in child modules.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "children"))
from features_block_1 import Model as FeaturesBlock1
from features_block_2 import Model as FeaturesBlock2
from features_block_3 import Model as FeaturesBlock3
from features_block_4 import Model as FeaturesBlock4
from features_block_5 import Model as FeaturesBlock5
from classifier import Model as Classifier

class RefactoredModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features_block_1 = FeaturesBlock1()
        self.features_block_2 = FeaturesBlock2()
        self.features_block_3 = FeaturesBlock3()
        self.features_block_4 = FeaturesBlock4()
        self.features_block_5 = FeaturesBlock5()
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        x = self.features_block_1(x)
        x = self.features_block_2(x)
        x = self.features_block_3(x)
        x = self.features_block_4(x)
        x = self.features_block_5(x)
        x = x.flatten(1)                  # data plumbing: shape op
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
