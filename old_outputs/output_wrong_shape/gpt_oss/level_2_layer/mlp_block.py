"""
GPT-OSS MoE MLP Block - Level 2 Layer Component

Component: MLPBlock
Level: 2 (Layer)
Parent: TransformerBlock (Level 2)
Children: MLPGating (L1), ExpertUpActivate (L1), ExpertDownCombine (L1)

Operations:
    1. RMSNorm on input
    2. Gate linear projection -> TopK expert selection -> Softmax weights
    3. Expert MLP1 (up-projection) via batched einsum
    4. SwiGLU activation with clamping
    5. Expert MLP2 (down-projection) via batched einsum
    6. Weighted expert combination + residual connection

Input Shape: x [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16
Output Shape: [SEQ_LEN=16, HIDDEN_SIZE=128] bfloat16

Test Configuration:
    SEQ_LEN=16, HIDDEN_SIZE=128, INTERMEDIATE_SIZE=128,
    NUM_EXPERTS=4, EXPERTS_PER_TOKEN=2, SWIGLU_LIMIT=7.0
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Mixture-of-Experts MLP block: RMSNorm -> Gating -> Expert Up+SwiGLU -> Expert Down+Combine + Residual.
    Combines all MLP-related operations into a single Level 2 layer.
    """

    def __init__(self, hidden_size=128, intermediate_size=128, num_experts=4,
                 experts_per_token=2, swiglu_limit=7.0):
        super().__init__()
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit

        # RMSNorm parameters
        self.norm_scale = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = 1e-5

        # Gate linear projection
        self.gate = nn.Linear(hidden_size, num_experts, dtype=torch.bfloat16)

        # MLP1 expert weights (up-projection with interleaved gate/linear for SwiGLU)
        self.mlp1_weight = nn.Parameter(
            torch.empty(num_experts, intermediate_size * 2, hidden_size, dtype=torch.bfloat16)
        )
        self.mlp1_bias = nn.Parameter(
            torch.empty(num_experts, intermediate_size * 2, dtype=torch.bfloat16)
        )

        # MLP2 expert weights (down-projection)
        self.mlp2_weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16)
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.bfloat16)
        )

        # Initialize
        nn.init.normal_(self.mlp1_weight, std=0.02)
        nn.init.zeros_(self.mlp1_bias)
        nn.init.normal_(self.mlp2_weight, std=0.02)
        nn.init.zeros_(self.mlp2_bias)

    def forward(self, x):
        """
        Forward pass for the MoE MLP block.

        Args:
            x: Input tensor [seq_len, hidden_size] bfloat16

        Returns:
            Output tensor [seq_len, hidden_size] bfloat16 (with residual connection)
        """
        # --- L1 Fusion: MLPGating ---
        # RMSNorm
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + self.eps)
        t = (t * self.norm_scale).to(x.dtype)

        # Gating: select top-k experts
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = F.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # --- L1 Fusion: ExpertUpActivate ---
        # Expert MLP1 (up-projection)
        mlp1_w = self.mlp1_weight[expert_indices, ...]  # [B, E, 2*I, H]
        mlp1_b = self.mlp1_bias[expert_indices, ...]      # [B, E, 2*I]
        h = torch.einsum("beck,bk->bec", mlp1_w, t) + mlp1_b

        # SwiGLU activation
        x_glu = h[..., ::2]
        x_linear = h[..., 1::2]
        x_glu = x_glu.clamp(min=None, max=self.swiglu_limit)
        x_linear = x_linear.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        out_glu = x_glu * torch.sigmoid(1.702 * x_glu)
        h = out_glu * (x_linear + 1)

        # --- L1 Fusion: ExpertDownCombine ---
        # Expert MLP2 (down-projection)
        mlp2_w = self.mlp2_weight[expert_indices, ...]  # [B, E, H, I]
        mlp2_b = self.mlp2_bias[expert_indices, ...]      # [B, E, H]
        h = torch.einsum("beck,bek->bec", mlp2_w, h)
        h += mlp2_b

        # Weighted expert combination
        h = torch.einsum("bec,be->bc", h, expert_weights)

        # Residual connection
        return x + h


def get_inputs():
    """Return sample inputs for testing."""
    return [torch.randn(16, 128, dtype=torch.bfloat16)]


def get_init_inputs():
    """Return constructor arguments for the Model."""
    return [128, 128, 4, 2, 7.0]


def get_expected_output_shape():
    """Return the expected output shape."""
    return [(16, 128)]


def run_tests():
    """Run validation tests for the MLPBlock component."""
    print("=" * 60)
    print("Testing: MLPBlock (Level 2 Layer)")
    print("=" * 60)

    passed = True

    # Test 1: Model instantiation
    try:
        model = Model(*get_init_inputs())
        model.eval()
        print("[PASS] Model instantiation")
    except Exception as e:
        print(f"[FAIL] Model instantiation: {e}")
        return False

    # Test 2: Forward pass shape
    try:
        inputs = get_inputs()
        with torch.no_grad():
            output = model(*inputs)
        expected_shapes = get_expected_output_shape()
        assert output.shape == torch.Size(expected_shapes[0]), (
            f"Expected shape {expected_shapes[0]}, got {output.shape}"
        )
        print(f"[PASS] Output shape: {output.shape}")
    except Exception as e:
        print(f"[FAIL] Forward pass shape: {e}")
        passed = False

    # Test 3: Output dtype
    try:
        assert output.dtype == torch.bfloat16, (
            f"Expected bfloat16, got {output.dtype}"
        )
        print(f"[PASS] Output dtype: {output.dtype}")
    except Exception as e:
        print(f"[FAIL] Output dtype: {e}")
        passed = False

    # Test 4: Output contains finite values
    try:
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        print("[PASS] Output values are finite")
    except Exception as e:
        print(f"[FAIL] Finite check: {e}")
        passed = False

    # Test 5: Residual connection check
    try:
        input_tensor = inputs[0]
        diff = (output - input_tensor).abs().max().item()
        assert diff > 0, "Output is identical to input - residual connection may be wrong"
        print(f"[PASS] Residual connection active (max diff: {diff:.4e})")
    except Exception as e:
        print(f"[FAIL] Residual connection: {e}")
        passed = False

    # Test 6: Deterministic output
    try:
        with torch.no_grad():
            output2 = model(*inputs)
        assert torch.allclose(output, output2, atol=1e-6), "Non-deterministic output"
        print("[PASS] Deterministic output")
    except Exception as e:
        print(f"[FAIL] Deterministic: {e}")
        passed = False

    # Test 7: Expert selection (verify top-k is working)
    try:
        with torch.no_grad():
            t = inputs[0].float()
            t = t * torch.rsqrt(torch.mean(t ** 2, dim=-1, keepdim=True) + model.eps)
            t = (t * model.norm_scale).to(inputs[0].dtype)
            g = model.gate(t)
            experts = torch.topk(g, k=model.experts_per_token, dim=-1, sorted=True)
            assert experts.indices.shape == (16, 2), (
                f"Expected expert indices shape (16, 2), got {experts.indices.shape}"
            )
        print(f"[PASS] Expert selection shape: {experts.indices.shape}")
    except Exception as e:
        print(f"[FAIL] Expert selection: {e}")
        passed = False

    print("=" * 60)
    print(f"MLPBlock: {'ALL TESTS PASSED' if passed else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return passed


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
