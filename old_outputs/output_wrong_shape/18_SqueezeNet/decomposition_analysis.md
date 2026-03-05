# SqueezeNet Hierarchical Decomposition Analysis

## Model: 18_SqueezeNet (KernelBench Level 3)

## Test Configuration
- **Batch size:** 2
- **Image size:** 32x32 (reduced from 224/512)
- **Num classes:** 10 (reduced from 1000)
- **Input shape:** [2, 3, 32, 32]
- **Output shape:** [2, 10]
- **Channel sizes:** Original (96, 128, 256, 384, 512) - kept full for correctness

## Architecture Overview

SqueezeNet uses "Fire Modules" as its core building block. Each Fire Module has:
1. **Squeeze layer:** 1x1 conv reducing channels (bottleneck)
2. **Expand layer:** Parallel 1x1 and 3x3 convolutions expanding channels
3. **Concatenation:** Channel-wise cat of both expand paths

The full model consists of:
- Initial convolution block (conv + relu + maxpool)
- 8 Fire Modules with 2 intermediate MaxPool layers
- Classifier (dropout + 1x1 conv + relu + adaptive avg pool + flatten)

## Decomposition Hierarchy

```
Level 3: Full SqueezeNet
  |
  +-- Level 2: Initial Block (Conv+ReLU+Pool)
  |     +-- Level 1: conv_relu_pool
  |           +-- Level 0: Conv2d(3,96,7,s=2)
  |           +-- Level 0: ReLU
  |           +-- Level 0: MaxPool2d(3,2,ceil)
  |
  +-- Level 2: Fire Module 1 (96->16->64+64=128)
  |     +-- Level 1: squeeze_relu
  |     |     +-- Level 0: Conv2d(96,16,1)
  |     |     +-- Level 0: ReLU
  |     +-- Level 1: expand_cat
  |           +-- Level 0: Conv2d(16,64,1) + ReLU  [parallel]
  |           +-- Level 0: Conv2d(16,64,3,p=1) + ReLU  [parallel]
  |           +-- Level 0: cat(dim=1)
  |
  +-- Level 2: Fire Module 2 (128->16->64+64=128)
  +-- Level 2: Fire Module 3 (128->32->128+128=256)
  +-- Level 0: MaxPool2d(3,2,ceil)
  +-- Level 2: Fire Module 4 (256->32->128+128=256)
  +-- Level 2: Fire Module 5 (256->48->192+192=384)
  +-- Level 2: Fire Module 6 (384->48->192+192=384)
  +-- Level 2: Fire Module 7 (384->64->256+256=512)
  +-- Level 0: MaxPool2d(3,2,ceil)
  +-- Level 2: Fire Module 8 (512->64->256+256=512)
  |
  +-- Level 2: Classifier Block
        +-- Level 0: Dropout(0.0)
        +-- Level 0: Conv2d(512,10,1)
        +-- Level 0: ReLU
        +-- Level 0: AdaptiveAvgPool2d(1,1)
        +-- Level 0: Flatten
```

## Shape Trace

| Step | Operation | Output Shape |
|------|-----------|-------------|
| Input | - | [2, 3, 32, 32] |
| Conv2d(3,96,7,s=2) | initial_block | [2, 96, 13, 13] |
| ReLU | initial_block | [2, 96, 13, 13] |
| MaxPool2d(3,2,ceil) | initial_block | [2, 96, 7, 7] |
| Fire1(96,16,64,64) | fire_module_1 | [2, 128, 7, 7] |
| Fire2(128,16,64,64) | fire_module_2 | [2, 128, 7, 7] |
| Fire3(128,32,128,128) | fire_module_3 | [2, 256, 7, 7] |
| MaxPool2d(3,2,ceil) | pool | [2, 256, 4, 4] |
| Fire4(256,32,128,128) | fire_module_4 | [2, 256, 4, 4] |
| Fire5(256,48,192,192) | fire_module_5 | [2, 384, 4, 4] |
| Fire6(384,48,192,192) | fire_module_6 | [2, 384, 4, 4] |
| Fire7(384,64,256,256) | fire_module_7 | [2, 512, 4, 4] |
| MaxPool2d(3,2,ceil) | pool | [2, 512, 2, 2] |
| Fire8(512,64,256,256) | fire_module_8 | [2, 512, 2, 2] |
| Dropout(0.0) | classifier_block | [2, 512, 2, 2] |
| Conv2d(512,10,1) | classifier_block | [2, 10, 2, 2] |
| ReLU | classifier_block | [2, 10, 2, 2] |
| AdaptiveAvgPool2d(1,1) | classifier_block | [2, 10, 1, 1] |
| Flatten | classifier_block | [2, 10] |

## File Inventory

### Level 3 (1 file)
- `level_3_model/squeezenet.py` - Full SqueezeNet model

### Level 2 (3 files)
- `level_2_layer/initial_block.py` - Initial conv+relu+pool
- `level_2_layer/fire_module_1.py` - Representative Fire Module (96->16->64+64)
- `level_2_layer/classifier_block.py` - Classifier head

### Level 1 (3 files)
- `level_1_fusion/conv_relu_pool.py` - Conv+ReLU+MaxPool fusion
- `level_1_fusion/squeeze_relu.py` - 1x1 squeeze conv+ReLU
- `level_1_fusion/expand_cat.py` - Parallel expand 1x1/3x3 + cat

### Level 0 (11 files)
- `level_0_kernel/conv2d_3_96.py` - Initial 7x7 convolution
- `level_0_kernel/relu.py` - ReLU activation
- `level_0_kernel/max_pool2d.py` - Max pooling
- `level_0_kernel/conv2d_squeeze.py` - 1x1 squeeze convolution
- `level_0_kernel/conv2d_expand1x1.py` - 1x1 expand convolution
- `level_0_kernel/conv2d_expand3x3.py` - 3x3 expand convolution
- `level_0_kernel/cat.py` - Channel concatenation
- `level_0_kernel/conv2d_classifier.py` - 1x1 classifier convolution
- `level_0_kernel/adaptive_avg_pool2d.py` - Global average pooling
- `level_0_kernel/dropout.py` - Dropout (p=0.0)
- `level_0_kernel/flatten.py` - Tensor flatten

### Verification (1 file)
- `verification/composition_test.py` - End-to-end equivalence tests

## Key Design Decisions

1. **Fire Modules are L2 layers:** Each Fire Module contains 7 operations (squeeze conv, relu, expand1x1 conv, relu, expand3x3 conv, relu, cat), making it a natural L2 layer.

2. **Representative decomposition:** Since all 8 Fire Modules share the same structure (only differing in channel counts), we provide one fully decomposed Fire Module (fire_module_1) as representative. The composition test verifies all 8.

3. **Parallel topology in expand_cat:** The expand path has a non-trivial topology - two parallel branches (1x1 and 3x3) that are concatenated. This is captured in the L1 expand_cat fusion.

4. **MaxPool layers between groups:** The MaxPool layers between fire module groups are standalone L0 operations (not fused into any L1/L2 block) since they serve as structural boundaries.

5. **Full channel sizes preserved:** We keep original channel dimensions (96, 128, 256, 384, 512) but use small spatial dimensions (32x32 input) to reduce computation while maintaining architectural fidelity.

## Verification

The composition test (`verification/composition_test.py`) verifies:
1. Single Fire Module equivalence (original vs composed)
2. Initial block equivalence
3. Classifier block equivalence
4. Full end-to-end model equivalence
5. All 18 intermediate tensor shapes
6. L0-to-L1 composition correctness
7. Parameter count matching

All tests use `torch.allclose(rtol=1e-4, atol=1e-5)` for numerical comparison.
