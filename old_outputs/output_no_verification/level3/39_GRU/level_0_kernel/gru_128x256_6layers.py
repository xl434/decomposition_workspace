"""
Component: GRU Kernel (128x256, 6 layers)
Source: data/kernelbench/level3/39_GRU.py
Abstraction Level: kernel
Parent: gru_fusion (level_1_fusion)
Operations: [nn.GRU forward pass - fused CUDA kernel containing reset/update gates, linear transforms, sigmoid, tanh, element-wise operations]
Input Shapes: x=[512, 10, 128], h0=[6, 10, 256]
Output Shapes: output=[512, 10, 256]

The GRU kernel is the atomic fused operation. PyTorch's nn.GRU executes as a single
fused CUDA kernel that internally performs all gate computations (reset gate, update gate,
new gate) across all 6 layers and all 512 time steps. This is the lowest-level decomposition
unit since the GRU is implemented as a monolithic fused operation in cuDNN/ATen.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        GRU kernel: nn.GRU(input_size=128, hidden_size=256, num_layers=6,
                           bias=True, batch_first=False, dropout=0, bidirectional=False)

        Args:
            input_size: Number of expected features in the input (128)
            hidden_size: Number of features in the hidden state (256)
            num_layers: Number of recurrent layers (6)
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False
        )

    def forward(self, x, h0):
        """
        Execute the GRU kernel.

        Args:
            x: Input tensor of shape [seq_len=512, batch_size=10, input_size=128]
            h0: Initial hidden state of shape [num_layers=6, batch_size=10, hidden_size=256]

        Returns:
            output: Output tensor of shape [seq_len=512, batch_size=10, hidden_size=256]
            h_n: Final hidden state of shape [num_layers=6, batch_size=10, hidden_size=256]
        """
        output, h_n = self.gru(x, h0)
        return output, h_n


def get_inputs():
    return [torch.randn(512, 10, 128), torch.randn(6, 10, 256)]


def get_init_inputs():
    return [128, 256, 6]


def get_expected_output_shape():
    return [(512, 10, 256), (6, 10, 256)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            # output is a tuple (output, h_n) at kernel level
            assert output is not None
            assert isinstance(output, tuple), f"Expected tuple, got {type(output)}"
            assert len(output) == 2, f"Expected 2 outputs, got {len(output)}"

            gru_output, h_n = output

            # Check output tensor
            assert not torch.isnan(gru_output).any(), "NaN in output"
            assert not torch.isinf(gru_output).any(), "Inf in output"
            assert tuple(gru_output.shape) == (512, 10, 256), \
                f"Output shape mismatch: {gru_output.shape} vs (512, 10, 256)"

            # Check hidden state tensor
            assert not torch.isnan(h_n).any(), "NaN in h_n"
            assert not torch.isinf(h_n).any(), "Inf in h_n"
            assert tuple(h_n.shape) == (6, 10, 256), \
                f"h_n shape mismatch: {h_n.shape} vs (6, 10, 256)"

            expected_shapes = get_expected_output_shape()
            actual_shapes = [gru_output.shape, h_n.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Shape mismatch at index {i}: {actual} vs {expected}"

            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {[tuple(s) for s in actual_shapes]}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
