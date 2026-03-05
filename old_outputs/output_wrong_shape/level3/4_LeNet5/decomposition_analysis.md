# LeNet-5 (4_LeNet5) Hierarchical Decomposition Analysis

## Model Overview

- **Source**: KernelBench Level 3, Model 4
- **Architecture**: LeNet-5 (classic CNN for digit classification)
- **Classification**: Level 2 (Layer) - 12 operations with clear sub-modules
- **Decomposition**: L2 Layer -> L1 Fusions -> L0 Kernels
- **Total Parameters**: 61,706

## Test Dimensions

| Parameter | Value |
|-----------|-------|
| batch_size | 2 |
| num_classes | 10 |
| input_shape | [2, 1, 32, 32] |
| output_shape | [2, 10] |

## Decomposition Tree

```
Level 2: lenet5 (full model)
  |
  +-- Level 1: conv_relu_pool_1 (3 ops)
  |     +-- L0: conv2d_1      Conv2d(1,6,5)    [2,1,32,32]  -> [2,6,28,28]
  |     +-- L0: relu_1        ReLU             [2,6,28,28]  -> [2,6,28,28]
  |     +-- L0: max_pool2d_1  MaxPool2d(2,2)   [2,6,28,28]  -> [2,6,14,14]
  |
  +-- Level 1: conv_relu_pool_2 (3 ops)
  |     +-- L0: conv2d_2      Conv2d(6,16,5)   [2,6,14,14]  -> [2,16,10,10]
  |     +-- L0: relu_2        ReLU             [2,16,10,10] -> [2,16,10,10]
  |     +-- L0: max_pool2d_2  MaxPool2d(2,2)   [2,16,10,10] -> [2,16,5,5]
  |
  +-- Level 1: flatten_fc_relu_1 (3 ops)
  |     +-- L0: flatten       View/Reshape     [2,16,5,5]   -> [2,400]
  |     +-- L0: linear_1      Linear(400,120)  [2,400]      -> [2,120]
  |     +-- L0: relu_3        ReLU             [2,120]      -> [2,120]
  |
  +-- Level 1: fc_relu_2 (2 ops)
  |     +-- L0: linear_2      Linear(120,84)   [2,120]      -> [2,84]
  |     +-- L0: relu_4        ReLU             [2,84]       -> [2,84]
  |
  +-- Level 1: fc_output (1 op)
        +-- L0: linear_3      Linear(84,10)    [2,84]       -> [2,10]
```

## Data Flow

```
Input [2,1,32,32]
  -> Conv2d(1,6,5)     -> [2,6,28,28]     (156 params)
  -> ReLU              -> [2,6,28,28]     (0 params)
  -> MaxPool2d(2,2)    -> [2,6,14,14]     (0 params)
  -> Conv2d(6,16,5)    -> [2,16,10,10]    (2,416 params)
  -> ReLU              -> [2,16,10,10]    (0 params)
  -> MaxPool2d(2,2)    -> [2,16,5,5]      (0 params)
  -> Flatten           -> [2,400]          (0 params)
  -> Linear(400,120)   -> [2,120]          (48,120 params)
  -> ReLU              -> [2,120]          (0 params)
  -> Linear(120,84)    -> [2,84]           (10,164 params)
  -> ReLU              -> [2,84]           (0 params)
  -> Linear(84,10)     -> [2,10]           (850 params)
Output [2,10]
```

## Parameter Distribution

| Layer | Parameters | Percentage |
|-------|-----------|------------|
| conv1 (Conv2d 1->6) | 156 | 0.25% |
| conv2 (Conv2d 6->16) | 2,416 | 3.92% |
| fc1 (Linear 400->120) | 48,120 | 77.98% |
| fc2 (Linear 120->84) | 10,164 | 16.47% |
| fc3 (Linear 84->10) | 850 | 1.38% |
| **Total** | **61,706** | **100%** |

## File Structure

```
output/level3/4_LeNet5/
  decomposition_tree.json
  decomposition_analysis.md
  level_0_kernel/
    conv2d_1.py
    relu_1.py
    max_pool2d_1.py
    conv2d_2.py
    relu_2.py
    max_pool2d_2.py
    flatten.py
    linear_1.py
    relu_3.py
    linear_2.py
    relu_4.py
    linear_3.py
  level_1_fusion/
    conv_relu_pool_1.py
    conv_relu_pool_2.py
    flatten_fc_relu_1.py
    fc_relu_2.py
    fc_output.py
  level_2_layer/
    lenet5.py
  verification/
    composition_test.py
```

## Fusion Rationale

1. **conv_relu_pool_1/2**: Classic Conv-ReLU-Pool pattern. These three operations are
   almost always fused in GPU implementations because ReLU is element-wise (can be
   fused with the conv output write) and pooling reads the same data just written.

2. **flatten_fc_relu_1**: The flatten is a zero-cost reshape that naturally pairs with
   the first FC layer. ReLU fuses trivially with the linear output.

3. **fc_relu_2**: Standard linear+activation fusion.

4. **fc_output**: Single operation (no activation on output layer). This is both an
   L1 fusion and an L0 kernel since it contains only one operation.

## Verification

The composition_test.py verifies:
- L0 kernels compose to match the original model (weight sharing, numerical equality)
- L1 fusions compose to match the original model
- L2 layer matches the original model
- Cross-level consistency (L0 == L1 == L2 == Original)
- All intermediate tensor shapes match the expected data flow
- Each individual module passes its own standalone tests
