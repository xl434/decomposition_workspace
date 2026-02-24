"""
End-to-End Composition Test for VGG16 Decomposition

Imports the original VGG16 model and all L0 kernel components,
builds a ComposedModel from ONLY kernel-level operations,
transfers weights, and verifies numerical equivalence.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import importlib.util

# Base paths
BASE_DIR = Path(__file__).parent.parent
KERNEL_DIR = BASE_DIR / "level_0_kernel"
ORIGINAL_DIR = BASE_DIR.parent.parent.parent / "data" / "kernelbench" / "level3"

# =========================================================================
# Load kernel modules
# =========================================================================

def load_model_class(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


# Load all L0 kernel classes
Conv2d_3x64_b1a = load_model_class(KERNEL_DIR / "conv2d_3x64_b1a_fp32.py", "k1")
ReLU_b1a = load_model_class(KERNEL_DIR / "relu_b1a_fp32.py", "k2")
Conv2d_64x64_b1b = load_model_class(KERNEL_DIR / "conv2d_64x64_b1b_fp32.py", "k3")
ReLU_b1b = load_model_class(KERNEL_DIR / "relu_b1b_fp32.py", "k4")
MaxPool2d_b1 = load_model_class(KERNEL_DIR / "maxpool2d_b1_fp32.py", "k5")

Conv2d_64x128_b2a = load_model_class(KERNEL_DIR / "conv2d_64x128_b2a_fp32.py", "k6")
ReLU_b2a = load_model_class(KERNEL_DIR / "relu_b2a_fp32.py", "k7")
Conv2d_128x128_b2b = load_model_class(KERNEL_DIR / "conv2d_128x128_b2b_fp32.py", "k8")
ReLU_b2b = load_model_class(KERNEL_DIR / "relu_b2b_fp32.py", "k9")
MaxPool2d_b2 = load_model_class(KERNEL_DIR / "maxpool2d_b2_fp32.py", "k10")

Conv2d_128x256_b3a = load_model_class(KERNEL_DIR / "conv2d_128x256_b3a_fp32.py", "k11")
ReLU_b3a = load_model_class(KERNEL_DIR / "relu_b3a_fp32.py", "k12")
Conv2d_256x256_b3b = load_model_class(KERNEL_DIR / "conv2d_256x256_b3b_fp32.py", "k13")
ReLU_b3b = load_model_class(KERNEL_DIR / "relu_b3b_fp32.py", "k14")
Conv2d_256x256_b3c = load_model_class(KERNEL_DIR / "conv2d_256x256_b3c_fp32.py", "k15")
ReLU_b3c = load_model_class(KERNEL_DIR / "relu_b3c_fp32.py", "k16")
MaxPool2d_b3 = load_model_class(KERNEL_DIR / "maxpool2d_b3_fp32.py", "k17")

Conv2d_256x512_b4a = load_model_class(KERNEL_DIR / "conv2d_256x512_b4a_fp32.py", "k18")
ReLU_b4a = load_model_class(KERNEL_DIR / "relu_b4a_fp32.py", "k19")
Conv2d_512x512_b4b = load_model_class(KERNEL_DIR / "conv2d_512x512_b4b_fp32.py", "k20")
ReLU_b4b = load_model_class(KERNEL_DIR / "relu_b4b_fp32.py", "k21")
Conv2d_512x512_b4c = load_model_class(KERNEL_DIR / "conv2d_512x512_b4c_fp32.py", "k22")
ReLU_b4c = load_model_class(KERNEL_DIR / "relu_b4c_fp32.py", "k23")
MaxPool2d_b4 = load_model_class(KERNEL_DIR / "maxpool2d_b4_fp32.py", "k24")

Conv2d_512x512_b5a = load_model_class(KERNEL_DIR / "conv2d_512x512_b5a_fp32.py", "k25")
ReLU_b5a = load_model_class(KERNEL_DIR / "relu_b5a_fp32.py", "k26")
Conv2d_512x512_b5b = load_model_class(KERNEL_DIR / "conv2d_512x512_b5b_fp32.py", "k27")
ReLU_b5b = load_model_class(KERNEL_DIR / "relu_b5b_fp32.py", "k28")
Conv2d_512x512_b5c = load_model_class(KERNEL_DIR / "conv2d_512x512_b5c_fp32.py", "k29")
ReLU_b5c = load_model_class(KERNEL_DIR / "relu_b5c_fp32.py", "k30")
MaxPool2d_b5 = load_model_class(KERNEL_DIR / "maxpool2d_b5_fp32.py", "k31")

Linear_25088x4096 = load_model_class(KERNEL_DIR / "linear_25088x4096_fp32.py", "k32")
ReLU_cls_a = load_model_class(KERNEL_DIR / "relu_cls_a_fp32.py", "k33")
Dropout_cls_a = load_model_class(KERNEL_DIR / "dropout_cls_a_fp32.py", "k34")
Linear_4096x4096 = load_model_class(KERNEL_DIR / "linear_4096x4096_fp32.py", "k35")
ReLU_cls_b = load_model_class(KERNEL_DIR / "relu_cls_b_fp32.py", "k36")
Dropout_cls_b = load_model_class(KERNEL_DIR / "dropout_cls_b_fp32.py", "k37")
Linear_4096x1000 = load_model_class(KERNEL_DIR / "linear_4096x1000_fp32.py", "k38")


# =========================================================================
# ComposedModel from L0 kernels only
# =========================================================================

class ComposedModel(nn.Module):
    """VGG16 composed entirely from L0 kernel components."""

    def __init__(self):
        super().__init__()
        # Block 1
        self.conv2d_b1a = Conv2d_3x64_b1a()
        self.relu_b1a = ReLU_b1a()
        self.conv2d_b1b = Conv2d_64x64_b1b()
        self.relu_b1b = ReLU_b1b()
        self.maxpool_b1 = MaxPool2d_b1()
        # Block 2
        self.conv2d_b2a = Conv2d_64x128_b2a()
        self.relu_b2a = ReLU_b2a()
        self.conv2d_b2b = Conv2d_128x128_b2b()
        self.relu_b2b = ReLU_b2b()
        self.maxpool_b2 = MaxPool2d_b2()
        # Block 3
        self.conv2d_b3a = Conv2d_128x256_b3a()
        self.relu_b3a = ReLU_b3a()
        self.conv2d_b3b = Conv2d_256x256_b3b()
        self.relu_b3b = ReLU_b3b()
        self.conv2d_b3c = Conv2d_256x256_b3c()
        self.relu_b3c = ReLU_b3c()
        self.maxpool_b3 = MaxPool2d_b3()
        # Block 4
        self.conv2d_b4a = Conv2d_256x512_b4a()
        self.relu_b4a = ReLU_b4a()
        self.conv2d_b4b = Conv2d_512x512_b4b()
        self.relu_b4b = ReLU_b4b()
        self.conv2d_b4c = Conv2d_512x512_b4c()
        self.relu_b4c = ReLU_b4c()
        self.maxpool_b4 = MaxPool2d_b4()
        # Block 5
        self.conv2d_b5a = Conv2d_512x512_b5a()
        self.relu_b5a = ReLU_b5a()
        self.conv2d_b5b = Conv2d_512x512_b5b()
        self.relu_b5b = ReLU_b5b()
        self.conv2d_b5c = Conv2d_512x512_b5c()
        self.relu_b5c = ReLU_b5c()
        self.maxpool_b5 = MaxPool2d_b5()
        # Classifier
        self.linear_1 = Linear_25088x4096()
        self.relu_cls_a = ReLU_cls_a()
        self.dropout_cls_a = Dropout_cls_a()
        self.linear_2 = Linear_4096x4096()
        self.relu_cls_b = ReLU_cls_b()
        self.dropout_cls_b = Dropout_cls_b()
        self.linear_3 = Linear_4096x1000()

    def forward(self, x):
        # Block 1
        x = self.conv2d_b1a(x)
        x = self.relu_b1a(x)
        x = self.conv2d_b1b(x)
        x = self.relu_b1b(x)
        x = self.maxpool_b1(x)
        # Block 2
        x = self.conv2d_b2a(x)
        x = self.relu_b2a(x)
        x = self.conv2d_b2b(x)
        x = self.relu_b2b(x)
        x = self.maxpool_b2(x)
        # Block 3
        x = self.conv2d_b3a(x)
        x = self.relu_b3a(x)
        x = self.conv2d_b3b(x)
        x = self.relu_b3b(x)
        x = self.conv2d_b3c(x)
        x = self.relu_b3c(x)
        x = self.maxpool_b3(x)
        # Block 4
        x = self.conv2d_b4a(x)
        x = self.relu_b4a(x)
        x = self.conv2d_b4b(x)
        x = self.relu_b4b(x)
        x = self.conv2d_b4c(x)
        x = self.relu_b4c(x)
        x = self.maxpool_b4(x)
        # Block 5
        x = self.conv2d_b5a(x)
        x = self.relu_b5a(x)
        x = self.conv2d_b5b(x)
        x = self.relu_b5b(x)
        x = self.conv2d_b5c(x)
        x = self.relu_b5c(x)
        x = self.maxpool_b5(x)
        # Flatten
        x = x.flatten(1)
        # Classifier
        x = self.linear_1(x)
        x = self.relu_cls_a(x)
        x = self.dropout_cls_a(x)
        x = self.linear_2(x)
        x = self.relu_cls_b(x)
        x = self.dropout_cls_b(x)
        x = self.linear_3(x)
        return x


# =========================================================================
# Weight transfer map: original state_dict key -> composed state_dict key
# =========================================================================

WEIGHT_MAP = {
    # Block 1
    "features.0.weight": "conv2d_b1a.op.weight",
    "features.0.bias": "conv2d_b1a.op.bias",
    "features.2.weight": "conv2d_b1b.op.weight",
    "features.2.bias": "conv2d_b1b.op.bias",
    # Block 2
    "features.5.weight": "conv2d_b2a.op.weight",
    "features.5.bias": "conv2d_b2a.op.bias",
    "features.7.weight": "conv2d_b2b.op.weight",
    "features.7.bias": "conv2d_b2b.op.bias",
    # Block 3
    "features.10.weight": "conv2d_b3a.op.weight",
    "features.10.bias": "conv2d_b3a.op.bias",
    "features.12.weight": "conv2d_b3b.op.weight",
    "features.12.bias": "conv2d_b3b.op.bias",
    "features.14.weight": "conv2d_b3c.op.weight",
    "features.14.bias": "conv2d_b3c.op.bias",
    # Block 4
    "features.17.weight": "conv2d_b4a.op.weight",
    "features.17.bias": "conv2d_b4a.op.bias",
    "features.19.weight": "conv2d_b4b.op.weight",
    "features.19.bias": "conv2d_b4b.op.bias",
    "features.21.weight": "conv2d_b4c.op.weight",
    "features.21.bias": "conv2d_b4c.op.bias",
    # Block 5
    "features.24.weight": "conv2d_b5a.op.weight",
    "features.24.bias": "conv2d_b5a.op.bias",
    "features.26.weight": "conv2d_b5b.op.weight",
    "features.26.bias": "conv2d_b5b.op.bias",
    "features.28.weight": "conv2d_b5c.op.weight",
    "features.28.bias": "conv2d_b5c.op.bias",
    # Classifier
    "classifier.0.weight": "linear_1.op.weight",
    "classifier.0.bias": "linear_1.op.bias",
    "classifier.3.weight": "linear_2.op.weight",
    "classifier.3.bias": "linear_2.op.bias",
    "classifier.6.weight": "linear_3.op.weight",
    "classifier.6.bias": "linear_3.op.bias",
}


def run_composition_test():
    print("=" * 60)
    print("END-TO-END COMPOSITION TEST")
    print("=" * 60)

    # Load original model
    print("\n[1/4] Loading original model...")
    orig_spec = importlib.util.spec_from_file_location(
        "original", str(ORIGINAL_DIR / "11_VGG16.py")
    )
    orig_mod = importlib.util.module_from_spec(orig_spec)
    orig_spec.loader.exec_module(orig_mod)

    original = orig_mod.Model(*orig_mod.get_init_inputs())
    original.eval()
    print(f"      Original params: {sum(p.numel() for p in original.parameters())}")

    # Create composed model
    print("[2/4] Creating composed model from L0 kernels...")
    composed = ComposedModel()
    composed.eval()
    print(f"      Composed params: {sum(p.numel() for p in composed.parameters())}")

    # Transfer weights
    print("[3/4] Transferring weights...")
    orig_sd = original.state_dict()
    comp_sd = composed.state_dict()

    for orig_key, comp_key in WEIGHT_MAP.items():
        if orig_key in orig_sd and comp_key in comp_sd:
            comp_sd[comp_key] = orig_sd[orig_key].clone()
        else:
            print(f"      [WARN] Missing mapping: {orig_key} -> {comp_key}")

    composed.load_state_dict(comp_sd)

    unmapped_orig = [k for k in orig_sd if k not in WEIGHT_MAP]
    unmapped_comp = [k for k in comp_sd if k not in WEIGHT_MAP.values()]
    if unmapped_orig:
        print(f"      [WARN] Unmapped original: {unmapped_orig}")
    if unmapped_comp:
        print(f"      [WARN] Unmapped composed: {unmapped_comp}")
    print(f"      Mapped {len(WEIGHT_MAP)} parameters")

    # Numerical comparison
    print("[4/4] Comparing outputs (3 trials)...")

    all_pass = True
    max_diff_overall = 0.0

    for trial in range(3):
        seed = 42 + trial
        torch.manual_seed(seed)
        inputs = orig_mod.get_inputs()

        with torch.no_grad():
            orig_out = original(*inputs)
            comp_out = composed(*inputs)

        diff = (orig_out.float() - comp_out.float()).abs().max().item()
        max_diff_overall = max(max_diff_overall, diff)
        matches = torch.allclose(orig_out.float(), comp_out.float(), rtol=1e-4, atol=1e-5)

        status = "PASS" if matches else "FAIL"
        print(f"      Trial {trial} (seed={seed}): {status} (max_diff={diff:.2e})")

        if not matches:
            all_pass = False

    print()
    print("-" * 60)
    if all_pass:
        print(f"[PASS] Composition test PASSED (max_diff={max_diff_overall:.2e})")
    else:
        print(f"[FAIL] Composition test FAILED (max_diff={max_diff_overall:.2e})")
    print("-" * 60)

    return all_pass


if __name__ == "__main__":
    success = run_composition_test()
    sys.exit(0 if success else 1)
