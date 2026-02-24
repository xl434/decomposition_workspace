"""
Refactored linear_relu_dropout_4096x4096_fp32: Fusion (L1) -> Kernels (L0)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "children"))
from linear_4096x4096_fp32 import Model as LinearKernel
from relu_cls_b_fp32 import Model as ReLUKernel
from dropout_cls_b_fp32 import Model as DropoutKernel

class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LinearKernel()
        self.relu = ReLUKernel()
        self.dropout = DropoutKernel()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

batch_size = 10

def get_inputs():
    return [torch.randn(batch_size, 4096)]

def get_init_inputs():
    return []
