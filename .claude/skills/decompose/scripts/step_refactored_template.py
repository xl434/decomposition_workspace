"""
Step Refactored Code Template

For each decomposition step (parent -> children), the agent creates a refactored
version of the parent module. This template shows the required structure.

RULES FOR REFACTORED CODE:
1. RefactoredModel.__init__() may ONLY contain child module instantiation
   - NO standalone nn.Linear, nn.Conv2d, nn.LayerNorm, etc.
   - All parameterized modules must come from imported child files
2. RefactoredModel.forward() may ONLY contain:
   - Calls to child modules: self.child_a(x)
   - Data plumbing: residual = x, x = x + residual, x = torch.cat([a, b], dim=1)
   - Shape ops: x.reshape(...), x.permute(...), x.transpose(...), x.view(...)
   - Indexing: x[:, :, 0], q, k, v = x.chunk(3, dim=-1)
   - NO: F.relu, F.softmax, torch.matmul, F.linear, or any compute ops
3. get_inputs() must return the SAME inputs as the parent component
4. get_init_inputs() must match the parent's initialization

VERIFICATION:
    python scripts/verify_step.py \\
        --original path/to/parent.py \\
        --refactored path/to/this_file.py \\
        --output path/to/verification_result.json

The verify_step.py script will:
- Check these rules (anti-cheat validation)
- Transfer weights from original to refactored
- Compare outputs numerically (must match within tolerance)
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# =========================================================================
# STEP INFO (fill in)
# =========================================================================
STEP_NAME = "{step_name}"  # e.g., "step_1_model_to_layers"
PARENT_FILE = "{parent_relative_path}"  # e.g., "level_3_model/vgg16.py"
CHILD_FILES = []  # e.g., ["children/features_block_1.py", "children/classifier.py"]

# =========================================================================
# IMPORTS: Only child modules
# =========================================================================
# Add the children directory to path
sys.path.insert(0, str(Path(__file__).parent / "children"))

# TODO: Import each child module
# from features_block_1 import Model as FeaturesBlock1
# from features_block_2 import Model as FeaturesBlock2
# from classifier import Model as Classifier


# =========================================================================
# REFACTORED MODEL
# =========================================================================

class RefactoredModel(nn.Module):
    """
    Refactored version of {parent_name}.

    This module's forward() calls child components instead of inline computation.
    All parameterized operations live in child modules.
    """

    def __init__(self):
        super().__init__()
        # TODO: ONLY instantiate imported child modules
        # self.features_block_1 = FeaturesBlock1()
        # self.features_block_2 = FeaturesBlock2()
        # self.classifier = Classifier()
        pass

    def forward(self, x):
        """
        Forward pass using ONLY child module calls and data plumbing.

        ALLOWED:
            - self.child(x)            # child module calls
            - x + residual             # residual connections
            - torch.cat([a, b], dim=1) # tensor assembly
            - x.reshape(B, S, H)       # shape manipulation
            - x[:, 0, :]              # indexing/slicing

        NOT ALLOWED:
            - F.relu(x), F.softmax(x)  # functional compute ops
            - torch.matmul(q, k)        # torch compute ops
            - nn.Linear(768, 768)(x)    # inline module creation
        """
        # TODO: Chain children in the same order as original's forward()
        #
        # Example for VGG16:
        # x = self.features_block_1(x)
        # x = self.features_block_2(x)
        # x = x.flatten(1)  # shape op: allowed
        # x = self.classifier(x)
        # return x
        #
        # Example for Transformer block with residuals:
        # residual = x                    # data plumbing: allowed
        # x = self.norm_attention(x)      # child call
        # x = x + residual               # residual add: allowed
        # residual = x                    # data plumbing: allowed
        # x = self.norm_mlp(x)           # child call
        # x = x + residual               # residual add: allowed
        # return x
        pass


# =========================================================================
# INPUT FUNCTIONS (must match parent)
# =========================================================================

def get_inputs():
    """
    Generate test inputs — MUST match the parent component's get_inputs().

    TODO: Copy from parent file.
    """
    # Example:
    # return [torch.randn(10, 3, 224, 224)]
    return []


def get_init_inputs():
    """Return initialization parameters — MUST match parent."""
    return []


# =========================================================================
# SELF-TEST (optional, for standalone testing)
# =========================================================================

if __name__ == "__main__":
    """
    Quick self-test. For full verification with weight transfer,
    use scripts/verify_step.py instead.
    """
    try:
        model = RefactoredModel(*get_init_inputs())
        model.eval()

        inputs = get_inputs()
        if not inputs:
            print("SKIP: No inputs defined yet")
            sys.exit(0)

        with torch.no_grad():
            output = model(*inputs)

        if isinstance(output, tuple):
            shapes = [o.shape for o in output if isinstance(o, torch.Tensor)]
        elif isinstance(output, torch.Tensor):
            shapes = [output.shape]
        else:
            shapes = [type(output)]

        print(f"Input shapes: {[x.shape for x in inputs if isinstance(x, torch.Tensor)]}")
        print(f"Output shapes: {shapes}")
        print("PASS (execution only — run verify_step.py for full verification)")
        sys.exit(0)

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
