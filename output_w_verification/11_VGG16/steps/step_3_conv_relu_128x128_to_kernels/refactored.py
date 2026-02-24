"""
Refactored conv_relu_128x128_fp32: Fusion (L1) -> Kernels (L0)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "children"))
from conv2d_128x128_b2b_fp32 import Model as Conv2dKernel
from relu_b2b_fp32 import Model as ReLUKernel

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = Conv2dKernel()
        self.relu = ReLUKernel()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        return x

batch_size = 10

def get_inputs():
    return [torch.randn(batch_size, 128, 112, 112)]

def get_init_inputs():
    return []
