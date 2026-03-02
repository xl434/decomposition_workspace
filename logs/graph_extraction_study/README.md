# Graph Extraction Study

Deploying PyTorch workloads to heterogeneous and emerging hardware requires
semantic understanding of model structure. Workload partitioning (deciding
which model components should run on which accelerator, e.g., vision encoder on
NPU, transformer on GPU) requires knowing the model's architectural
boundaries. Kernel selection (matching each operation to the best available
implementation across potentially scattered toolchains) requires knowing what
each operation is (a GELU MLP, a grouped-query attention block), not just its
low-level arithmetic.

Existing compiler-based methods for extracting computation graphs such as
`torch.fx`, `torch.compile`, and `torch.export` do not provide this.
They suffer from two categories of limitations. First, graph extraction itself
is fragile: `torch.fx` and `torch.export` fail outright on architecturally
complex models due to data-dependent control flow, dynamic tensor shapes, and
framework-specific object access patterns. Even `torch.compile`, the most
robust method, fragments these models into disconnected subgraphs at graph
breaks. Second, even when extraction succeeds, the resulting flat ATen-level
graph erases semantic structure, so there is no way to determine which of the
560 operations in GPT-2's traced graph correspond to the attention mechanism
versus the feed-forward network, let alone identify partition boundaries for
heterogeneous deployment.

Monolithic compilers like TorchInductor sidestep this problem by tightly
coupling graph extraction with a fixed backend (CUDA/Triton), but this approach
does not generalize: each new accelerator platform requires writing a new
compiler backend, and the semantic information lost during extraction cannot be
recovered downstream. Our agentic decomposition takes a different approach,
producing a hierarchical, semantically structured, verified representation that
is independent of any target hardware.

This study conducts a systematic empirical study across 9 models spanning the
KernelBench Level 4 benchmark and the SmolVLA vision-language-action model to
quantify the limitations of compiler-based extraction.


## Experimental Setup

Evaluate four graph extraction methods on models of increasing architectural
complexity:

- **`torch.fx.symbolic_trace`**: Static symbolic tracing that replaces tensor
  inputs with `Proxy` objects and records operations.
- **`torch.compile`**: JIT compilation via TorchDynamo that captures ATen-level
  operations, tolerating graph breaks where tracing falls back to eager
  execution.
- **`torch.export`**: Strict whole-program capture that produces a single,
  self-contained graph with no graph breaks permitted.
- **Forward hooks**: Runtime interception of `nn.Module` forward calls on leaf
  modules.

The test suite covers 7 unique architectures from KernelBench Level 4
(GPT-2, GPT-Neo, OPT, BigBird, BART, ELECTRA, Reformer) plus SmolVLA, a
multimodal VLA model combining a SigLIP vision encoder, SmolVLM language
backbone, and an action expert with flow matching. All models are instantiated
with `from_config` (random weights) to isolate architectural effects from weight
initialization. Experiments run on PyTorch 2.7.1 (CPU).


## Results

**Table 1.** Graph extraction results across methods. "FAIL" indicates the method
could not produce any graph. Graph breaks (*gb*) indicate where `torch.compile`
splits the computation into separate subgraphs.

| Model                  | torch.fx | torch.compile     | torch.export | Hooks    |
|------------------------|----------|-------------------|--------------|----------|
| GPT-2 (124M)          | FAIL     | 560 ops           | 575 ops      | 113 ops  |
| GPT-Neo (125M)        | FAIL     | 521 ops           | 608 ops      | 149 ops  |
| OPT (125M)            | FAIL     | 345 ops           | 370 ops      | 112 ops  |
| BigBird (127M)        | FAIL     | 449 ops           | 456 ops      | 151 ops  |
| BART (139M)           | FAIL     | 460 ops           | 463 ops      | 145 ops  |
| ELECTRA (13M)         | FAIL     | 363 ops           | 368 ops      | 150 ops  |
| Reformer (149M)       | FAIL     | 683 ops (6 gb)    | FAIL         | FAIL     |
| SmolVLA (244M)        | FAIL     | 3083 ops (7 gb)   | FAIL         | 440 ops  |
| SmolVLM backbone (256M) | FAIL   | 2309 ops (11 gb)  | FAIL         | 416 ops  |
| **Success rate**       | **0/9**  | **9/9**           | **6/9**      | **8/9**  |

Several patterns emerge:

### torch.fx fails on all models (0/9)

Every HuggingFace model contains input validation logic of the form:

```python
if input_ids is not None and inputs_embeds is not None:
    raise ValueError("Cannot specify both...")
```

When `symbolic_trace` passes `Proxy` objects, both arguments appear
non-`None`, triggering the guard. This is a fundamental incompatibility: any
model with optional-argument branching is untraceable by `torch.fx`.

### torch.export fails on complex architectures (3/9 failures)

`torch.export` requires a single, complete, graph-break-free trace. It fails on
Reformer (data-dependent branching from `np.lcm` in LSH attention), SmolVLA
(data-dependent `torch.arange` in the vision encoder), and the SmolVLM backbone
(same vision encoder issue). These are not corner cases as they represent common
patterns in production models.

### torch.compile always succeeds but fragments the graph

While `torch.compile` handles all 9 models, it produces graph breaks on 3 of
them, with SmolVLA being the most severe at 7 graph breaks splitting the model
into 8 separate subgraphs. We analyze this case in detail below.

### Forward hooks miss functional operations

Hooks only capture `nn.Module` calls, missing all functional operations
(`F.relu`, `torch.matmul`, tensor arithmetic, masking). For GPT-2, hooks see
113 module-level ops while `torch.compile` captures 560 ATen-level ops, a
5.0x gap. For SmolVLA, the ratio is 7.0x (440 vs. 3083 ops).


---

## Operator Granularity: What torch.compile Actually Captures

Beyond graph breaks and tracing failures, a subtler limitation of compiler-based
extraction is the inconsistent abstraction level of the captured operations.
To understand what `torch.compile` and `torch.export` actually produce, we
examine the complete op breakdown for GPT-2 (124M), which compiles cleanly into
a single graph with no graph breaks.

**Table 5.** ATen operator breakdown for GPT-2 under `torch.compile` (560 total
ops, 23 unique op types). Operations are grouped by category.

| Category                  | Ops  | % of total | Key operators                            |
|---------------------------|------|------------|------------------------------------------|
| Shape/view manipulation   | 269  | 48.0%      | `view` (135), `contiguous` (60), `transpose` (48), `reshape` (12), `split` (12), `expand` (1), `unsqueeze` (1) |
| Elementwise arithmetic    | 110  | 19.6%      | `add` (50), `mul` (48), `pow` (12)       |
| Indexing / getitem        |  49  |  8.8%      | `getitem` (49)                           |
| Matrix multiply           |  49  |  8.8%      | `addmm` (48), `linear` (1)              |
| Normalization             |  25  |  4.5%      | `layer_norm` (25)                        |
| Regularization            |  25  |  4.5%      | `dropout` (25)                           |
| Fused attention           |  12  |  2.1%      | `scaled_dot_product_attention` (12)      |
| Activation (decomposed)   |  12  |  2.1%      | `tanh` (12)                              |
| Other                     |   9  |  1.6%      | `embedding` (2), `arange` (2), `to` (2), `full` (1), `lt` (1), `masked_fill_` (1) |

Several observations reveal the inconsistent abstraction level:

### Some operations are kept as high-level kernels

`layer_norm` appears as a single ATen op, called 25 times (once per transformer
sublayer). Each invocation corresponds to one `nn.LayerNorm` module. Similarly,
`scaled_dot_product_attention` (SDPA) appears 12 times (once per attention
head) as a single fused op that internally computes Q·K^T scaling, masking,
softmax, and the V projection. As a result, there is no `softmax` op anywhere
in the graph as it is hidden inside SDPA.

### Other operations are decomposed into primitives

In contrast, GPT-2's GELU activation (`gelu_new`, the tanh approximation) is
not preserved as a single op. Instead, the computation:

```python
# gelu_new: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

is decomposed into 4 separate ATen ops per layer:

```
pow  ->  mul  ->  add  ->  tanh  ->  mul  ->  add  ->  mul
x^3    coeff    x+ax^3   tanh()    scale    1+t()   0.5*x*()
```

This produces 12 `tanh` ops + 12 `pow` ops and contributes to the `mul` and
`add` counts. Looking at the captured graph, there is no `gelu` op. The
activation function has been shattered into arithmetic primitives, making it
unrecognizable without manual analysis.

Critically, this decomposition is not a limitation of the ATen operator set
itself as ATen does include a native `gelu` op. The issue is that
HuggingFace's `gelu_new` activation (`NewGELUActivation` in
`transformers/activations.py`) implements the tanh approximation as explicit
arithmetic (`torch.tanh`, `torch.pow`, `*`, `+`) rather than calling
`torch.nn.functional.gelu(x, approximate='tanh')`. HuggingFace even provides
a `PytorchGELUTanh` wrapper that does call `F.gelu` and would appear as a
single ATen op, but GPT-2's config defaults to `gelu_new`. Thus, whether the
same mathematical function appears as 1 op or 7 ops in the compiled graph
depends on an implementation detail chosen by the model author, a detail
invisible to downstream users trying to map workloads to hardware.

### Shape manipulation dominates the graph

Nearly half (48%) of all captured ops are shape manipulations: `view`,
`contiguous`, `transpose`, `reshape`, `split`, `expand`, `unsqueeze`. These
carry no computational meaning as they exist only because PyTorch's ATen
backend requires explicit tensor layout management. For downstream tasks like
hardware mapping or kernel fusion, these ops are noise that must be filtered.

### torch.export shows the same pattern

`torch.export` produces a nearly identical op distribution (575 ops vs. 560 for
`torch.compile`, same 23+ unique op types). Both keep `layer_norm` and SDPA as
single ops while decomposing GELU. The inconsistency is inherent to the ATen
operator set, not a quirk of one tracing method.

### Comparison: consistent abstraction in hierarchical decomposition

Our decomposition avoids this inconsistency by design. At each level of the
hierarchy, operations are represented at a uniform abstraction:

| Level | Abstraction | Example: attention block             | Example: GELU MLP block                |
|-------|-------------|--------------------------------------|-----------------------------------------|
| 2     | Layer       | "Transformer Layer"                  | "Transformer Layer"                     |
| 1     | Fusion      | "SDPA" (one named component)         | "GELU MLP" (one named component)        |
| 0     | Kernel      | `matmul`, `softmax`, `matmul`        | `linear`, `gelu`, `linear`              |

At Level 1, both SDPA and GELU MLP are single named components with documented
I/O shapes. At Level 0, both are decomposed into their constituent kernels
(`gelu` remains a single kernel op, not 4 arithmetic primitives). A user or
tool operating at Level 1 sees uniform fused blocks; at Level 0, they see
uniform kernel primitives. There is no level where `layer_norm` is one op but
`gelu` is four.


---

## Case Study: SmolVLA Graph Fragmentation

SmolVLA is a particularly instructive example because it combines multiple
modalities (vision, language, proprioception, action) with flow matching for
action denoising, cross-attention between a VLM backbone and an action expert,
and sinusoidal time embeddings. We use `torch._dynamo.explain()` to identify
the root causes of its 7 graph breaks.

### Root Cause 1: Data-dependent output shapes in the vision encoder

The SigLIP vision embedding computes per-image position encodings using:

```python
nb_patches_h = p_attn_mask[:, 0].sum()
fractional_coords_h = torch.arange(
    0, 1 - 1e-6, 1 / nb_patches_h)   # GRAPH BREAK
```

Here `nb_patches_h` is a runtime tensor value derived from the input attention
mask. Since the output shape of `torch.arange` depends on this value,
TorchDynamo inserts a graph break at the `aten._local_scalar_dense` operator
that converts the tensor to a Python scalar. This pattern appears in a per-batch
loop, generating 3–4 fragment subgraphs for the vision embedding phase alone.

This is also the precise reason `torch.export` fails entirely: it cannot
symbolically guard on the expression `zuf0 > 0.0` (the positivity of the step
size), as this depends on input data.

### Root Cause 2: Data-dependent control flow in the vision model

```python
if not torch.any(~patch_attention_mask):  # GRAPH BREAK
    patch_attention_mask = None
```

TorchDynamo classifies this as a fundamental graph break, one that "is
unlikely [to] ever be able to trace through." The branch condition depends on
the content of an input tensor, which cannot be resolved at trace time.

All 7 graph breaks trace back to these two root causes: repeated instances of
`aten._local_scalar_dense` (data-dependent `arange`) and data-dependent
control flow in the vision encoder and its connector to the language backbone.
Notably, the VLM+Expert transformer itself (16 cross-attention layers with
RoPE embeddings and SwiGLU MLPs) compiles into a single clean 2762-op
subgraph with no internal graph breaks, demonstrating that the fragmentation
is concentrated in the vision preprocessing pipeline rather than the
transformer backbone.

### Graph fragmentation summary

**Table 2.** SmolVLA subgraph breakdown under `torch.compile`. The model is
split into 8 fragments; semantic mapping to model components is lost.

| Graph | Ops   | Component (inferred)       | Break cause                            |
|-------|-------|----------------------------|----------------------------------------|
| 1     | 6     | Vision embedding (patch 1) | Data-dependent `arange`                |
| 2     | 1     | Vision embedding (patch 2) | Data-dependent `arange`                |
| 3     | 2     | Vision embedding (patch 3) | Data-dependent `arange`                |
| 4     | 3     | Vision embedding (patch 4) | Data-dependent control flow            |
| 5     | 301   | SigLIP encoder             | (clean - no break within)              |
| 6     | 7     | Vision-language connector  | Data-dependent `arange` / control flow |
| 7     | 1     | Projection transition      | Data-dependent `arange` / control flow |
| 8     | 2762  | VLM+Expert + output        | (clean - no break within)              |
| **Total** | **3083** |                       | **7 graph breaks**                     |


---

## Comparison with Hierarchical Decomposition

Our decomposition approach operates at the module level rather than the
operator level, which fundamentally sidesteps the two failure modes above.

**Table 3.** Comparison of graph extraction approaches on SmolVLA.

| Property                        | torch.fx | torch.compile        | torch.export | Agentic Decomposition                  |
|---------------------------------|----------|----------------------|--------------|-----------------------------------------|
| Succeeds?                       | No       | Yes                  | No           | Yes                                     |
| Output fragments                | —        | 8 subgraphs          | —            | 30 components                           |
| Abstraction levels              | —        | 1 (ATen ops)         | —            | 4 (model/layer/fusion/kernel)           |
| Semantic structure preserved    | —        | No                   | —            | Yes                                     |
| Handles data-dep. control flow  | No       | Partial (breaks)     | No           | Yes                                     |
| Consistent op abstraction       | —        | No (`gelu` → 4 ops) | —            | Yes (uniform per level)                 |
| Verified composability          | —        | —                    | —            | Yes (15/15 steps, max diff = 0.0)       |

Concretely, our decomposition of SmolVLA produces a 4-level hierarchy with 30
named components, each with verified input/output specifications. All 15
decomposition steps pass numerical verification with a maximum difference of
0.0 (bitwise-exact composition).

**Table 4.** Hierarchical decomposition of SmolVLA: 30 components across 4
levels. Instance counts shown where a component is reused.

| Level               | Component              | Instances | I/O shape                           |
|---------------------|------------------------|-----------|-------------------------------------|
| *Level 3 - Model*   |                        |           |                                     |
|                     | VLAFlowMatching        | 1         | multi-modal → [1,50,32]             |
| *Level 2 - Layer*   |                        |           |                                     |
|                     | Vision Encoder (SigLIP)| 1         | [1,3,512,512] → [1,64,960]          |
|                     | Action-Time MLP        | 1         | [1,50,32]+[1] → [1,50,720]          |
|                     | VLM+Expert Transformer | 1         | [1,113,960]+[1,50,720] → [1,50,720] |
| *Level 1 - Fusion*  |                        |           |                                     |
|                     | Vision Patch Embed     | 1         | [1,3,512,512] → [1,1024,768]        |
|                     | Vision SDPA            | 12        | [1,1024,768] → [1,1024,768]         |
|                     | Vision GELU MLP        | 12        | [1,1024,768] → [1,1024,768]         |
|                     | Vision LayerNorm       | 25        | [1,1024,768] → [1,1024,768]         |
|                     | Pixel Shuffle Proj     | 1         | [1,1024,768] → [1,64,960]           |
|                     | Sinusoidal Pos Emb     | 1         | [1] → [1,720]                       |
|                     | SiLU Linear MLP        | 1         | [1,50,1440] → [1,50,720]            |
|                     | RMS Norm               | 66        | [1,113,960] → [1,113,960]           |
|                     | Grouped Query Attention| 1         | QKV+mask → [1,163,960]              |
|                     | RoPE Embedding         | 1         | [1,163,15,64] → [1,163,15,64]       |
|                     | SwiGLU MLP             | 32        | [1,113,960] → [1,113,960]           |
| *Level 0 - Kernel*  |                        |           |                                     |
|                     | linear, conv2d, matmul, softmax, gelu, silu, embedding, layer_norm | 8 types | |


## Key Advantages of Agentic Decomposition

### 1. Robustness to dynamic patterns

Because decomposition works with the module hierarchy (identifying components
like "Vision Encoder" or "Grouped Query Attention" at the architectural level),
it does not need to trace through data-dependent control flow or `torch.arange`
with dynamic step sizes. These patterns cause fundamental failures in
`torch.fx` and `torch.export`, and fragment the graph under `torch.compile`.

This robustness is critical for heterogeneous deployment: if graph extraction
fails on a model, it simply cannot be deployed to an emerging accelerator at
all. SmolVLA is exactly the kind of complex workload that would benefit from
disaggregated execution across heterogeneous hardware. Yet `torch.fx` and
`torch.export` fail entirely on it, and `torch.compile` shatters it into 8
graph fragments. The agentic approach is the promising method that produces a
usable structural representation for all 9 models tested.

### 2. Preserved semantic structure enables kernel selection and workload partitioning

The 8 subgraphs produced by `torch.compile` are anonymous ATen-level
fragments: there is no information about which fragment corresponds to the
vision encoder, the action expert, or the attention mechanism. In contrast, our
decomposition produces named, hierarchically organized components with
documented I/O specifications.

This semantic structure directly enables the two downstream tasks central to our
agentic compiler:

- **Kernel selection**: Knowing that a component is "GELU MLP" or "Grouped
  Query Attention" rather than an anonymous sequence of `pow`, `mul`, `add`,
  `tanh` allows the agent to search for accelerator-specific implementations
  by semantic name. This is the "active search beyond a small internal set of
  implementations" described in our system: the agent can query kernel libraries,
  vendor documentation, and code repositories using the component's
  architectural identity, not just its op signature.

- **Disaggregated deployment**: Named Layer-level components (vision encoder,
  language backbone, action expert) provide natural partition boundaries for
  routing to different accelerators. For SmolVLA, the decomposition immediately
  reveals that the vision encoder (SigLIP, [1,3,512,512] → [1,64,960]) and the
  VLM+Expert transformer ([1,113,960]+[1,50,720] → [1,50,720]) are independent
  pipeline stages that could be mapped to different devices (e.g., NPU and GPU).
  The anonymous ATen fragments from `torch.compile` provide no such guidance.

### 3. Multi-level abstraction with consistent granularity

Compiler-based methods produce a single-level graph of low-level operations
(ATen ops for `torch.compile`/`export`; module calls for hooks). As shown in
the operator granularity analysis above, this single level has inconsistent
granularity: `layer_norm` and `scaled_dot_product_attention` are preserved as
single ops, while `gelu` is decomposed into four arithmetic primitives, and
48% of the graph is shape manipulation noise (`view`, `contiguous`,
`transpose`). Our decomposition provides four abstraction levels, each with
uniform granularity from the whole model (Level 3) down to individual kernels
(Level 0).

Each level could map to a concrete role in the agentic compiler pipeline:

- **Level 2 (Layer)**: Disaggregated deployment partitioning, e.g., vision
  encoder on NPU, transformer backbone on GPU, action expert on a dedicated
  accelerator.
- **Level 1 (Fusion)**: The unit at which the agent searches for or writes
  kernel implementations. Each fusion (e.g., "SwiGLU MLP", "SDPA") is a
  self-contained compute block with well-defined I/O, suitable for matching
  against accelerator-specific kernel libraries.
- **Level 0 (Kernel)**: Standalone benchmarkable units that can be individually
  profiled on each target accelerator, enabling hardware profiling-guided
  selection. A `linear_768x3072_fp32` kernel can be profiled on GPU, NPU, and
  any emerging accelerator independently.

### 4. Verified composability as a foundation for kernel replacement

Each decomposition step is accompanied by a numerical verification test that
composes the child components and checks that the result matches the parent's
output. Crucially, this verification infrastructure is reusable beyond the
initial decomposition. When the agentic compiler replaces a kernel with an
accelerator-specific implementation (a Triton kernel, a vendor-provided NPU
op, or a custom-written kernel), the same verification framework can confirm
that the replacement is numerically correct. The standalone, self-testing nature
of each decomposed component means that kernel substitution can be validated
at the component level without re-running the entire model.


---

## Discussion

Our results do not argue that `torch.compile` is ineffective, instead it
successfully captures the computation for all 9 models and remains the best
tool for JIT optimization on supported platforms. Rather, we argue that for
the purpose of deploying workloads to heterogeneous and emerging hardware,
compiler-based graph extraction is insufficient on its own.

The fundamental limitation is that compiler-based methods operate bottom-up:
tracing individual operations and attempting to assemble them into a graph. Our
top-down decomposition follows the opposite direction: starting from the
model's module hierarchy and recursively splitting into semantically meaningful
components, each verified by a unit test that composes the children and checks
agreement with the parent. This mirrors standard software engineering practice:
decompose, implement, test.
