# Exercise: diagnose CUDA memory empirically

This 50–60 minute exercise uses phase-by-phase measurements to distinguish
activation pressure, persistent optimizer state, and an accidentally retained
training graph. The cases are labelled `alpha`, `beta`, and `gamma` so the
diagnosis must come from evidence rather than the option name.

[`memory_diagnostics.py`](memory_diagnostics.py) creates a synthetic transformer
and GPU-resident inputs locally. It downloads no model, weights, tokenizer, or
dataset. One GPU is sufficient. An optional DDP extension compares ranks on a
multi-GPU node.

## Learning objectives and success criteria

After the exercise, participants should be able to:

1. distinguish live tensor allocation from memory reserved by PyTorch's caching
   allocator;
2. identify whether a memory increase occurs during forward propagation, on the
   first optimizer step, or cumulatively between training steps;
3. test a diagnosis by changing one workload dimension at a time;
4. choose a targeted intervention and reject an unrelated one.

The exercise is complete when the participant correctly diagnoses all three
cases, cites a measured phase transition or scaling trend for each diagnosis,
and proposes a matching response. Producing the JSON files alone is not the
success criterion.

Required prerequisites are familiarity with a PyTorch training cycle and a
CUDA-enabled PyTorch environment with at least one GPU. The repository's
[`environment_gpu.yml`](../../../environment_gpu.yml) provides the intended
software environment. DDP knowledge and a second GPU are optional.

## Files

| File | Purpose |
|---|---|
| [`memory_diagnostics.py`](memory_diagnostics.py) | Runs an anonymous workload, records memory at model, input, forward, backward, update, and gradient-clearing boundaries, and writes JSON. |
| [`compare_memory_reports.py`](compare_memory_reports.py) | Produces a compact comparison table from one or more JSON reports without naming the diagnosis. |
| [`SOLUTION.md`](SOLUTION.md) | Complete reference procedure, expected diagnoses, reasoning, and limitations. Consult it after forming a hypothesis. |

## What is measured

Every phase reports four related but non-interchangeable quantities:

| Metric | Interpretation |
|---|---|
| `allocated_mib` | Live tensor storage known to PyTorch's CUDA allocator. Use this first when reasoning about tensor lifetimes. |
| `reserved_mib` | Blocks retained by the caching allocator. Released tensors can reduce allocated memory without immediately reducing this value. |
| `peak_allocated_mib` | Largest live PyTorch allocation since the peak was reset at the start of the current step. |
| `device_used_mib` | Device-wide consumption derived from free memory. It includes CUDA libraries and other processes, so an exclusive Slurm allocation is preferable. |

The report also derives three diagnosis aids:

- `first_step_persistent_change_mib`: allocation remaining after the first step
  and gradient clearing, relative to the pre-training baseline;
- `between_step_growth_mib`: change in the post-cleanup baseline from the first
  to the final step;
- `largest_forward_increase_mib`: the largest forward allocation above the
  preceding post-cleanup baseline.

These indicators are evidence, not universal thresholds. Absolute numbers vary
with GPU, precision, PyTorch version, kernel selection, and model shape.

## Part 1: predict and collect a baseline

Before running, sketch the expected live-allocation curve for one training
step. Mark model construction, forward propagation, backward propagation, the
first optimizer update, and gradient clearing.

Run all three cases with the same configuration:

```bash
python memory_diagnostics.py --case alpha --output alpha.json
python memory_diagnostics.py --case beta --output beta.json
python memory_diagnostics.py --case gamma --output gamma.json

python compare_memory_reports.py --show-phases alpha.json beta.json gamma.json
```

For each case, answer these questions before looking at the implementation:

1. Is its main change transient within a step, persistent after the first step,
   or cumulative across steps?
2. Which phase provides the strongest evidence?
3. Is the gap between allocated and reserved memory itself evidence of a leak?

Do not interpret a large `reserved_mib` value alone as live memory pressure.

## Part 2: test each hypothesis

A diagnosis needs a perturbation, not just a plausible story. Change one
variable at a time.

### Batch-size sweep

Predict which indicator will respond most strongly, then run:

```bash
python memory_diagnostics.py --case alpha --batch-size-per-device 1 \
    --output alpha-batch1.json
python memory_diagnostics.py --case alpha --batch-size-per-device 4 \
    --output alpha-batch4.json

python compare_memory_reports.py alpha-batch1.json alpha-batch4.json
```

### Model-size sweep

Predict whether transient and persistent allocations will change in the same
proportion:

```bash
python memory_diagnostics.py --case beta --preset small \
    --output beta-small.json
python memory_diagnostics.py --case beta --preset medium \
    --batch-size-per-device 1 --output beta-medium.json

python compare_memory_reports.py beta-small.json beta-medium.json
```

### Step-count sweep

Run the suspected cumulative case for longer. Compare the *post-cleanup
baseline*, not only its largest peak:

```bash
python memory_diagnostics.py --case gamma --steps 10 \
    --output gamma-steps10.json
```

If `medium` exceeds the assigned GPU or takes too long, omit that run. An OOM is
capacity evidence, not a required outcome and not a valid performance result.

## Part 3: recommend an intervention

For every case, choose one primary response and explain why at least one other
response is poorly targeted. Candidate responses include:

- reduce the local microbatch;
- use activation checkpointing;
- shard model state with FSDP;
- change optimizer or optimizer-state precision;
- stop retaining training graphs and store detached CPU summaries instead.

The question is not “which technique saves memory?” but “which technique acts
on the measured source of pressure?”

## Optional detailed snapshot

Counters should answer the first question. Once one run is suspicious, collect
a detailed allocator history:

```bash
python memory_diagnostics.py --case gamma --steps 6 \
    --snapshot gamma-snapshot.pickle \
    --output gamma-with-snapshot.json
```

The generated file is rank-qualified, for example
`gamma-snapshot-rank0.pickle`. Copy it from the compute node and open it in the
[PyTorch memory visualizer](https://pytorch.org/memory_viz). Recording histories
adds overhead and can create large files, which is why it is optional and
bounded by `--snapshot-max-entries`.

PyTorch allocator snapshots cannot see allocations made directly by NCCL and
some other CUDA libraries. Compare snapshot evidence with `device_used_mib`
rather than assuming the snapshot accounts for the entire device.

## Optional DDP rank-imbalance extension

Use two GPUs first with equal local batches, then deliberately give rank zero a
larger batch:

```bash
torchrun --standalone --nproc-per-node=2 memory_diagnostics.py \
    --case alpha --output alpha-ddp-balanced.json

torchrun --standalone --nproc-per-node=2 memory_diagnostics.py \
    --case alpha --rank-zero-extra-samples 2 \
    --output alpha-ddp-imbalanced.json
```

Inspect every entry in `rank_reports`. Rank-zero-only logging would hide a
problem on another rank in a real workload. The artificial imbalance also
changes the effective weighting of samples in ordinary DDP gradient averaging;
it is a diagnostic demonstration, not a recommended training configuration.

### Slurm example

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00

set -euo pipefail

srun torchrun --standalone --nproc-per-node=2 memory_diagnostics.py \
    --case alpha \
    --output "alpha-ddp-${SLURM_JOB_ID}.json"
```

Adapt the GPU directive, partition, account, and environment activation to the
local cluster. One Slurm task launches `torchrun`; `torchrun` creates one worker
per assigned GPU.

## Hints

1. Many optimizers create state lazily during their first `step()`, not when the
   optimizer object is constructed.
2. A healthy steady-state training loop normally reuses memory rather than
   increasing live allocation on every iteration.
3. `empty_cache()` does not release live tensors. Do not add it to conceal an
   unexplained growth trend.
4. `detach()` breaks an autograd connection but does not move or discard the
   underlying GPU tensor. For logging, retain a scalar or transfer the required
   result to CPU.

## Core code map

Function and class names are more durable than line numbers if the file changes.

| Area | Function or class and line | What to inspect |
|---|---|---|
| Options and validation | [`parse_arguments()` — line 103](memory_diagnostics.py#L103) | Defines the case, workload, precision, output, snapshot, and imbalance options. |
| CUDA/DDP setup | [`initialize_runtime()` — line 184](memory_diagnostics.py#L184) | Selects one GPU and initializes NCCL only when launched by `torchrun`. |
| Model blocks | [`TransformerBlock` — line 236](memory_diagnostics.py#L236) | Defines the attention and feed-forward unit. |
| Complete model | [`SyntheticTransformer` — line 267](memory_diagnostics.py#L267) | Creates the transformer used by every case. |
| Model/DDP creation | [`make_model()` — line 294](memory_diagnostics.py#L294) | Constructs an identical local model and optionally wraps it in DDP. |
| Input handling | [`make_batch()` — line 322](memory_diagnostics.py#L322) | Creates rank-specific synthetic token batches directly on the GPU. |
| Memory sampling | [`sample_memory()` — line 347](memory_diagnostics.py#L347) | Records allocated, reserved, peak, and device-wide memory at a phase boundary. |
| Training cycle | [`training_step()` — line 362](memory_diagnostics.py#L362) | Measures forward, backward, optimizer, and cleanup phases. |
| Evidence derivation | [`derive_indicators()` — line 425](memory_diagnostics.py#L425) | Converts raw samples into three hypothesis-testing indicators. |
| Orchestration | [`run()` — line 467](memory_diagnostics.py#L467) | Builds the workload, collects evidence, optionally dumps snapshots, and gathers ranks. |

## Limitations

- Synchronizing at every phase makes the measurement boundaries clear but
  distorts step timing; this is a memory experiment, not a throughput benchmark.
- Synthetic reused input excludes data-loader and host-to-device transfer
  memory. Diagnose those separately in a real application.
- Model-size changes affect both parameters and activations. The small-batch
  medium run reduces, but does not eliminate, that confounder.
- DDP may create persistent gradient-bucket storage during its first backward
  pass. Diagnose the three core cases on one GPU first; in the extension,
  compare otherwise identical DDP runs rather than applying single-GPU
  thresholds blindly.
- The anonymous cases isolate one dominant cause each. Real workloads can have
  several simultaneous sources of pressure.
- Allocator internals used for optional snapshots have underscored names and
  are more version-sensitive than the ordinary counter API.

## Further reading

- [PyTorch CUDA memory management API](https://docs.pytorch.org/docs/stable/cuda.html#memory-management)
- [PyTorch: Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [PyTorch `max_memory_allocated`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)
