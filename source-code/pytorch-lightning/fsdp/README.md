# PyTorch Lightning FSDP2 benchmark

[`fsdp2_transformer.py`](fsdp2_transformer.py) is the PyTorch Lightning
counterpart of the
[native PyTorch DDP/FSDP2 benchmark](../../pytorch/fsdp/fsdp2_transformer.py).
It uses the same synthetic transformer, presets, optimizer, batch modes,
warm-up period, timed steps, and result metrics.

No model, weights, tokenizer, or dataset are downloaded. The model is assembled
from standard PyTorch modules, and every rank creates a distinct synthetic token
batch directly on its GPU.

This is primarily a memory-scaling exercise. FSDP2 exchanges parameters during
the forward and backward passes so that parameters, gradients, and optimizer
state do not need to remain fully replicated. That usually reduces per-GPU
model-state memory, but it can make training slower than DDP when the complete
model already fits comfortably.

## Requirements

The script targets the Lightning 2.6 `ModelParallelStrategy` API and requires
PyTorch 2.4 or newer. The repository's
[`environment_gpu.yml`](../../../environment_gpu.yml) selects PyTorch 2.13 but
does not install Lightning, because Lightning is an optional dependency:

```bash
python -m pip install lightning
```

Install it in a project-specific environment on a cluster rather than changing
a shared system Python installation. The example needs a CUDA-enabled PyTorch
build and at least two GPUs to demonstrate sharding; a one-GPU run is useful
only as a smoke test.

The use of
[`ModelParallelStrategy`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.ModelParallelStrategy.html)
is deliberate. Lightning's older `FSDPStrategy` wraps PyTorch's FSDP1 class.
`ModelParallelStrategy` exposes the FSDP2 and device-mesh workflow, and this
script sets `tensor_parallel_size=1` so that every worker participates only in
the data-parallel/FSDP dimension. Lightning currently labels this strategy
experimental, so version upgrades may require small API changes. PyTorch's
[FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
explains the underlying per-parameter sharding model.

## Model presets

Start with `small`. Validate `medium` on the actual lab hardware before the
session. `large` can exhaust smaller GPUs and is best kept as an optional
capacity experiment.

| Preset | Blocks | Hidden size | Parameters | FP32 parameters only | Approximate DDP model state with AdamW |
|---|---:|---:|---:|---:|---:|
| `small` | 12 | 768 | 88,327,680 | 0.33 GiB | 1.32 GiB |
| `medium` | 24 | 1,024 | 306,563,072 | 1.14 GiB | 4.57 GiB |
| `large` | 32 | 1,280 | 634,903,040 | 2.37 GiB | 9.46 GiB |

The final column estimates FP32 parameters, gradients, and the two AdamW
moment tensors after the first optimizer step. It excludes activations,
communication buckets, all-gather buffers, CUDA context memory, allocator
fragmentation, and NCCL allocations. Actual peak memory will therefore be
higher.

## Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--strategy {ddp,fsdp2}` | required | Select replicated Lightning DDP or Lightning-managed FSDP2. |
| `--preset {small,medium,large}` | `small` | Select model dimensions from the table above. |
| `--fsdp-wrap {block,root}` | `block` | Choose per-block or root-only FSDP2 grouping; ignored for DDP. |
| `--batch-size-per-device N` | `1` | Set the local batch; keeping it fixed gives weak scaling. |
| `--global-batch-size N` | unset | Set a fixed total batch for strong scaling; mutually exclusive with the local batch option. |
| `--warmup-steps N` | `5` | Run untimed steps that initialize optimizer state and communication paths. |
| `--steps N` | `15` | Set the number of timed optimizer steps. |
| `--precision {auto,fp32,bf16}` | `auto` | Use BF16 when the selected GPU supports it; otherwise use FP32. |
| `--seed N` | `1234` | Control model initialization and rank-specific synthetic inputs. |
| `--devices N` | `1` | Set GPUs per node; under Slurm this must match the tasks per node. |
| `--num-nodes N` | `1` | Set the Lightning/Slurm node count. |
| `--output PATH` | unset | Write the rank-zero result as JSON; existing files are protected. |
| `--overwrite-output` | off | Permit replacement of an existing result file. |

The script validates numbers, batch divisibility, and output behavior before
starting the Trainer. Each result includes the resolved configuration, seed,
framework versions, throughput, peak PyTorch CUDA memory, and parameter
checksums.

## Core code map

The line numbers below refer to the current version of
[`fsdp2_transformer.py`](fsdp2_transformer.py). Function and method names are
the more durable reference if the file changes later.

| Area | Function or method and line | What to inspect |
|---|---|---|
| Options and validation | [`parse_arguments()` — line 97](fsdp2_transformer.py#L97) | Defines strategy, preset, wrapping, batch, precision, device, timing, and output options. |
| Precision setup | [`resolve_precision()` — line 183](fsdp2_transformer.py#L183) | Resolves FP32 or BF16 before Lightning launches workers. |
| Transformer block | [`TransformerBlock` — line 199](fsdp2_transformer.py#L199) | Defines the unit used as an FSDP2 communication and sharding boundary. |
| Complete model | [`SyntheticTransformer` — line 230](fsdp2_transformer.py#L230) | Defines embeddings, transformer blocks, normalization, and classifier. |
| Model creation and FSDP2 setup | [`TransformerBenchmark.configure_model()` — line 304](fsdp2_transformer.py#L304) | Creates the large layers in Lightning's strategy-aware hook, then applies `fully_shard()` to blocks and root. |
| Distributed batch interpretation | [`TransformerBenchmark.setup()` — line 348](fsdp2_transformer.py#L348) | Uses Lightning's actual world size to resolve local and global batches. |
| Synthetic input | [`TransformerBenchmark.on_train_start()` — line 368](fsdp2_transformer.py#L368) | Measures the local parameter shard and creates rank-specific GPU-resident tokens. |
| One training cycle | [`TransformerBenchmark.training_step()` — line 392](fsdp2_transformer.py#L392) | Runs forward propagation and loss calculation; Lightning performs backward, synchronization, and update. |
| Optimizer setup | [`TransformerBenchmark.configure_optimizers()` — line 402](fsdp2_transformer.py#L402) | Creates AdamW after `configure_model()` has sharded the model. |
| Training-loop input | [`TransformerBenchmark.train_dataloader()` — line 409](fsdp2_transformer.py#L409) | Supplies one lightweight token per optimizer step; the actual token tensor is already on the GPU. |
| Warm-up and measured cycles | [`on_train_batch_start()` — line 415](fsdp2_transformer.py#L415) and [`on_train_batch_end()` — line 425](fsdp2_transformer.py#L425) | Synchronize workers, exclude warm-up, time the measured steps, and collect peak CUDA memory. |
| Result construction | [`TransformerBenchmark.on_train_end()` — line 473](fsdp2_transformer.py#L473) | Reduces correctness and performance metrics and writes rank-zero output. |
| DDP or FSDP2 strategy | [`make_strategy()` — line 579](fsdp2_transformer.py#L579) | Selects `DDPStrategy` or a pure-FSDP2 `ModelParallelStrategy` device mesh. |
| Trainer setup | [`main()` — line 594](fsdp2_transformer.py#L594) | Configures Lightning's GPU devices, nodes, precision, strategy, and barebones training loop. |

Start with `main()`, then inspect `configure_model()` and `training_step()`.
The key FSDP2 detail is that large layers are created in the strategy-aware
hook, sharding is applied before optimizer construction, and the model code
itself remains ordinary PyTorch.

## What Lightning changes

| Responsibility | Native PyTorch example | Lightning example |
|---|---|---|
| Process group, ranks, and teardown | Explicit NCCL setup and cleanup | `Trainer` and the selected strategy |
| Device placement | Explicit local-rank CUDA binding | Lightning accelerator |
| DDP wrapping | Explicit `DistributedDataParallel(...)` | `DDPStrategy` |
| FSDP2 device mesh | Implicit world process group | `ModelParallelStrategy` creates and exposes the mesh |
| FSDP2 wrapping policy | Explicit `fully_shard()` calls | Still explicit in `configure_model()` because layer boundaries are a model decision |
| Mixed precision | DDP autocast or FSDP2 policy | Trainer precision for DDP; `MixedPrecisionPolicy` for FSDP2 communication and parameters |
| Backward and optimizer step | Explicit calls in `training_step()` | Automatic after the Lightning step returns loss |
| Timing and reporting | Custom benchmark code | Still custom, because these are benchmark requirements |

This is intentionally an exact comparison rather than the shortest possible
Lightning program. Input checks, timing, JSON reporting, and correctness checks
therefore still account for much of the file. The simplification is clearest in
`main()` and `training_step()`.

## Run the comparison

Use the same GPU count, preset, precision, and batch settings for both runs. On
a four-GPU node outside Slurm:

```bash
python fsdp2_transformer.py \
    --strategy ddp \
    --devices 4 \
    --preset medium \
    --batch-size-per-device 1 \
    --output result-medium-ddp.json

python fsdp2_transformer.py \
    --strategy fsdp2 \
    --devices 4 \
    --fsdp-wrap block \
    --preset medium \
    --batch-size-per-device 1 \
    --output result-medium-fsdp2-block.json
```

If `medium` is too slow or exceeds memory, substitute `small` in both commands.
Changing only one workload invalidates the comparison. For strong scaling,
replace the local batch option in both commands with a global batch divisible
by the worker count, for example `--global-batch-size 4`.

## Run as a single-node Slurm batch job

Lightning detects the Slurm environment. Unlike the native example, which
runs one Slurm launcher task and lets `torchrun` create workers, this version
uses one Slurm task per GPU. The Slurm task count, GPU allocation, and
`--devices` value must agree:

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

set -euo pipefail

srun python fsdp2_transformer.py \
    --strategy ddp \
    --devices 4 \
    --preset medium \
    --output result-medium-ddp.json

srun python fsdp2_transformer.py \
    --strategy fsdp2 \
    --devices 4 \
    --fsdp-wrap block \
    --preset medium \
    --output result-medium-fsdp2-block.json
```

Some clusters spell the GPU request as `#SBATCH --gres=gpu:4`; follow the local
site documentation. For multi-node runs, set both the Slurm node count and
`--num-nodes`, and keep `--devices` equal to the number of tasks and GPUs on
each node.

## Hands-on tasks

Before running, predict which strategy will use less peak memory and which will
have higher throughput. Then:

1. Run DDP and block-wrapped FSDP2 with otherwise identical options.
2. Compare `local_parameter_fraction`, `peak_allocated_gib`,
   `milliseconds_per_step`, and `tokens_per_second` in the JSON results.
3. Explain why DDP stores nearly all parameters on every rank while FSDP2
   approaches a local fraction of `1 / world_size`.
4. Repeat FSDP2 with `--fsdp-wrap root`; predict its peak memory and throughput
   before looking at the result.
5. Compare the Lightning script with the native implementation and identify
   which distributed responsibilities disappeared and which model-specific
   decisions remained.
6. If hardware and time permit, find the largest preset each strategy can run.
   Treat an out-of-memory outcome as capacity evidence, not throughput data.

## Interpret the output carefully

- `local_parameter_fraction` measures parameter elements stored by a rank. It
  excludes gradients, optimizer state, activations, and communication buffers.
- FSDP2 peak memory does not fall exactly by `1 / world_size`: the active layer
  is gathered for computation, activations remain local, and CUDA libraries
  need temporary memory.
- Block wrapping permits layer-wise gathering and potential communication/
  computation overlap. Root-only wrapping gathers the whole model as one unit
  and is included as a deliberately poor boundary choice.
- `parameters_synchronized` is meaningful for replicated DDP parameters. It is
  `null` for FSDP2 because ranks hold complementary shards; global checksums are
  reported instead.
- `--precision auto` is resolved before Lightning starts distributed workers
  and therefore assumes homogeneous GPUs. Use an explicit precision on a
  heterogeneous allocation.
- `barebones=True` disables logging, checkpoints, progress bars, and other
  Trainer services that would distort a short benchmark. It is not a sensible
  default for a production training application.
- Peak memory uses PyTorch's CUDA allocator counters and does not capture every
  allocation made directly by NCCL or other CUDA libraries.
- Synthetic, reused input isolates model and communication costs. The results
  are not end-to-end training performance and say nothing about model accuracy.
- The seed makes model initialization and inputs repeatable, but GPU kernels
  are not forced into deterministic implementations.

Activation checkpointing and distributed checkpoint files are intentionally
left out. They are useful follow-ups, but adding either would introduce a second
memory mechanism and obscure the first DDP/FSDP2 comparison.
The separate
[`activation-checkpointing`](../activation-checkpointing/) exercise isolates
that memory/recomputation trade-off before participants combine techniques.
