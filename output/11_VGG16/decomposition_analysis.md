# VGG16 Hierarchical Decomposition Analysis

## Model Overview

VGG16 is a classic convolutional neural network with 16 weight layers (13 Conv2d + 3 Linear).

- **Input**: [10, 3, 224, 224] float32 (batch of 10 RGB images at 224x224)
- **Output**: [10, 1000] float32 (1000-class logits)
- **Total Parameters**: 138,357,544

## Architecture Hierarchy

```
VGG16 (L3 - Model)
├── Features Block 1 (L2 - Layer): [10,3,224,224] -> [10,64,112,112]
│   ├── Conv2d(3,64)+ReLU (L1 - Fusion)
│   │   ├── Conv2d(3,64,3,p=1) (L0 - Kernel)
│   │   └── ReLU (L0 - Kernel)
│   ├── Conv2d(64,64)+ReLU (L1 - Fusion)
│   │   ├── Conv2d(64,64,3,p=1) (L0 - Kernel)
│   │   └── ReLU (L0 - Kernel)
│   └── MaxPool2d(2,2) (L1/L0 - single op)
├── Features Block 2 (L2 - Layer): [10,64,112,112] -> [10,128,56,56]
│   ├── Conv2d(64,128)+ReLU (L1)
│   ├── Conv2d(128,128)+ReLU (L1)
│   └── MaxPool2d(2,2) (L1/L0)
├── Features Block 3 (L2 - Layer): [10,128,56,56] -> [10,256,28,28]
│   ├── Conv2d(128,256)+ReLU (L1)
│   ├── Conv2d(256,256)+ReLU (L1) x2
│   └── MaxPool2d(2,2) (L1/L0)
├── Features Block 4 (L2 - Layer): [10,256,28,28] -> [10,512,14,14]
│   ├── Conv2d(256,512)+ReLU (L1)
│   ├── Conv2d(512,512)+ReLU (L1) x2
│   └── MaxPool2d(2,2) (L1/L0)
├── Features Block 5 (L2 - Layer): [10,512,14,14] -> [10,512,7,7]
│   ├── Conv2d(512,512)+ReLU (L1) x3
│   └── MaxPool2d(2,2) (L1/L0)
├── [flatten] (data plumbing): [10,512,7,7] -> [10,25088]
└── Classifier (L2 - Layer): [10,25088] -> [10,1000]
    ├── Linear(25088,4096)+ReLU+Dropout (L1)
    │   ├── Linear(25088,4096) (L0)
    │   ├── ReLU (L0)
    │   └── Dropout(p=0.0) (L0)
    ├── Linear(4096,4096)+ReLU+Dropout (L1)
    │   ├── Linear(4096,4096) (L0)
    │   ├── ReLU (L0)
    │   └── Dropout(p=0.0) (L0)
    └── Linear(4096,1000) (L1/L0 - single op)
```

## Component Count

| Level | Count |
|-------|-------|
| L3 Model | 1 |
| L2 Layer | 6 |
| L1 Fusion | 21 |
| L0 Kernel | 38 |
| **Total** | **66** |

## Kernel Breakdown

| Kernel Type | Count |
|-------------|-------|
| Conv2d | 13 |
| ReLU | 15 |
| MaxPool2d | 5 |
| Linear | 3 |
| Dropout | 2 |
| **Total** | **38** |

## Verification Summary

### Step-by-Step Verification (All PASSED)

| Step | Parent -> Children | Status | max_diff |
|------|-------------------|--------|----------|
| step_1_model_to_layers | VGG16 -> 5 blocks + classifier | PASS | 0.00e+00 |
| step_2_block1_to_fusions | Block1 -> 2 ConvReLU + MaxPool | PASS | 0.00e+00 |
| step_2_block2_to_fusions | Block2 -> 2 ConvReLU + MaxPool | PASS | 0.00e+00 |
| step_2_block3_to_fusions | Block3 -> 3 ConvReLU + MaxPool | PASS | 0.00e+00 |
| step_2_block4_to_fusions | Block4 -> 3 ConvReLU + MaxPool | PASS | 0.00e+00 |
| step_2_block5_to_fusions | Block5 -> 3 ConvReLU + MaxPool | PASS | 0.00e+00 |
| step_2_classifier_to_fusions | Classifier -> 2 LRD + Linear | PASS | 0.00e+00 |
| step_3_conv_relu_* (x13) | ConvReLU -> Conv2d + ReLU | ALL PASS | 0.00e+00 |
| step_3_linear_relu_dropout_* (x2) | LRD -> Linear + ReLU + Dropout | ALL PASS | 0.00e+00 |

### End-to-End Composition Test: PASS (max_diff=0.00e+00)

### Coverage: 97.4% (missing only `flatten` which is a shape op, not a compute kernel)

## Decomposition Decisions

1. **Model -> Layers**: Natural split along VGG16's 5 convolutional blocks + classifier
2. **Layers -> Fusions**: Conv+ReLU pairs as fusions (commonly fused in hardware); MaxPool2d standalone
3. **Fusions -> Kernels**: Each fusion split into individual atomic operations
4. **flatten**: Kept as data plumbing in the refactored forward(), not extracted as a kernel (it's a shape operation with no parameters)
5. **Dropout(p=0.0)**: Preserved as explicit kernels despite being no-ops at eval time, to maintain structural fidelity to the original model
