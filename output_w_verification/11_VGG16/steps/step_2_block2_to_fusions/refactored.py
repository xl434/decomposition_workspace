"""
Refactored Features Block 2: Layer (L2) -> Fusions (L1)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "children"))
from conv_relu_64x128_fp32 import Model as ConvRelu1
from conv_relu_128x128_fp32 import Model as ConvRelu2
from maxpool2d_b2_fp32 import Model as MaxPool

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_1 = ConvRelu1()
        self.conv_relu_2 = ConvRelu2()
        self.maxpool = MaxPool()

    def forward(self, x):
        x = self.conv_relu_1(x)
        x = self.conv_relu_2(x)
        x = self.maxpool(x)
        return x

batch_size = 10

def get_inputs():
    return [torch.randn(batch_size, 64, 112, 112)]

def get_init_inputs():
    return []
