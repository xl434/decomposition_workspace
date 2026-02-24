"""
Refactored Classifier: Layer (L2) -> Fusions (L1)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "children"))
from linear_relu_dropout_25088x4096_fp32 import Model as LinearReLUDropout1
from linear_relu_dropout_4096x4096_fp32 import Model as LinearReLUDropout2
from linear_4096x1000_fp32 import Model as FinalLinear

class RefactoredModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.linear_relu_dropout_1 = LinearReLUDropout1()
        self.linear_relu_dropout_2 = LinearReLUDropout2()
        self.final_linear = FinalLinear()

    def forward(self, x):
        x = self.linear_relu_dropout_1(x)
        x = self.linear_relu_dropout_2(x)
        x = self.final_linear(x)
        return x

batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 512 * 7 * 7)]

def get_init_inputs():
    return [num_classes]
