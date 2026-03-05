"""
Component: Embedding lookup
Abstraction Level: kernel
Operations: nn.Embedding lookup

Input Shapes:
  - indices: [B, S] int64

Output Shapes:
  - out: [B, S, embed_dim] float32

Weight Shapes:
  - weight: [num_embeddings, embed_dim]
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Embedding lookup kernel."""
    def __init__(self, num_embeddings=1024, embedding_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, indices):
        return self.embedding(indices)


def get_inputs():
    return [torch.randint(0, 1024, (1, 64))]

def get_init_inputs():
    return [1024, 768]

def get_expected_output_shape():
    return [(1, 64, 768)]

def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected = get_expected_output_shape()
            assert tuple(output.shape) == tuple(expected[0]), \
                f"Shape mismatch: {output.shape} vs {expected[0]}"
            print(f"Output shape: {output.shape}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
