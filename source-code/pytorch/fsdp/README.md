# PyTorch FSDP2 benchmark

[`fsdp2_transformer.py`](fsdp2_transformer.py) compares native PyTorch
`DistributedDataParallel` (DDP) with Fully Sharded Data Parallel version 2
(FSDP2). It uses the same synthetic transformer workload for both strategies
and reports per-GPU memory, throughput, timing, and parameter-storage metrics.

No model, weights, tokenizer, or dataset are downloaded. The model is an
encoder-style transformer assembled from standard PyTorch modules, and each
rank creates a distinct synthetic token batch directly on its GPU.

This is primarily a memory-scaling exercise. FSDP2 may be slower than DDP when
the complete model already fits comfortably on every GPU. That is not a failed
experiment: it demonstrates the cost of exchanging parameters in return for
lower persistent model-state memory.

## Learning objectives and prerequisites

After this 45–60 minute hands-on, participants should be able to:

1. distinguish DDP replication from FSDP parameter, gradient, and optimizer
   sharding;
2. compare memory and throughput without changing the model or effective
   batch;
3. explain why FSDP peak memory does not decrease by exactly `1 / world_size`;
4. choose a transformer-block sharding boundary instead of blindly sharding
   only the root model.

Required prerequisites are the preceding DDP exercise, basic PyTorch training
knowledge, a single node with at least two CUDA GPUs, and a CUDA-enabled build
of PyTorch 2.13 as specified by this repository's
[`environment_gpu.yml`](../../../environment_gpu.yml).
BF16 support is helpful but not required: `--precision auto` falls back to FP32
when any participating GPU lacks BF16 support.

The example uses the public `torch.distributed.fsdp.fully_shard` API. PyTorch's
[FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
marks FSDP1 as deprecated and explains the current per-parameter design.

## Model presets

Start with `small`. The instructor should validate `medium` on the actual lab
hardware before the session. `large` is deliberately capable of exhausting
smaller GPUs and should be treated as an optional capacity experiment.

| Preset | Blocks | Hidden size | Parameters | FP32 parameters only | Approximate DDP model state with AdamW |
|---|---:|---:|---:|---:|---:|
| `small` | 12 | 768 | 88,327,680 | 0.33 GiB | 1.32 GiB |
| `medium` | 24 | 1,024 | 306,563,072 | 1.14 GiB | 4.57 GiB |
| `large` | 32 | 1,280 | 634,903,040 | 2.37 GiB | 9.46 GiB |

The final column estimates FP32 parameters, gradients, and the two AdamW
moment tensors after the first optimizer step. It excludes activations, DDP
buckets, temporary all-gather buffers, CUDA context memory, allocator
fragmentation, and NCCL allocations, so actual peak memory will be higher.

## Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--strategy {ddp,fsdp2}` | required | Select replicated DDP or fully sharded training. |
| `--preset {small,medium,large}` | `small` | Select the model dimensions from the table above. |
| `--fsdp-wrap {block,root}` | `block` | Choose per-transformer-block or root-only FSDP grouping; ignored for DDP. |
| `--batch-size-per-device N` | `1` | Set the local batch; keeping it fixed gives a weak-scaling comparison. |
| `--global-batch-size N` | unset | Set a fixed total batch for strong scaling; mutually exclusive with the local batch option. |
| `--warmup-steps N` | `5` | Run untimed steps that initialize optimizer state and communication paths. |
| `--steps N` | `15` | Set the number of timed optimizer steps. |
| `--precision {auto,fp32,bf16}` | `auto` | Use BF16 only when all ranks support it; `auto` otherwise selects FP32. |
| `--seed N` | `1234` | Control model initialization and rank-specific synthetic inputs. |
| `--output PATH` | unset | Write the rank-zero result as JSON; existing files are protected. |
| `--overwrite-output` | off | Permit replacement of an existing `--output` file. |

## Core code map

The line numbers below refer to the current version of
[`fsdp2_transformer.py`](fsdp2_transformer.py). Function and class names are
the more durable reference if the file changes later.

| Area | Function or class and line | What to inspect |
|---|---|---|
| Options and validation | [`parse_arguments()` — line 92](fsdp2_transformer.py#L92) | Defines the strategy, preset, wrapping, batch, precision, timing, and output options. |
| Distributed setup | [`initialize_distributed()` — line 162](fsdp2_transformer.py#L162) | Reads `torchrun` variables, binds one GPU per process, and initializes NCCL. |
| Batch interpretation | [`resolve_batch_sizes()` — line 194](fsdp2_transformer.py#L194) | Distinguishes fixed per-device and fixed global batches. |
| Precision setup | [`resolve_precision()` — line 214](fsdp2_transformer.py#L214) | Selects BF16 only when every participating GPU supports it. |
| Transformer block | [`TransformerBlock` — line 234](fsdp2_transformer.py#L234) | Defines the unit used as an FSDP communication and sharding boundary. |
| Complete model | [`SyntheticTransformer` — line 265](fsdp2_transformer.py#L265) | Creates embeddings, transformer blocks, normalization, and a classification head. |
| Synthetic input | [`make_batch()` — line 292](fsdp2_transformer.py#L292) | Creates distinct GPU-resident tokens and labels for each rank. |
| DDP or FSDP2 setup | [`configure_distributed_model()` — line 319](fsdp2_transformer.py#L319) | Wraps the full model in DDP or applies `fully_shard()` to each block and then the root. |
| Local shard measurement | [`local_parameter_elements()` — line 353](fsdp2_transformer.py#L353) | Counts parameter elements physically stored on one rank. |
| One training cycle | [`training_step()` — line 365](fsdp2_transformer.py#L365) | Runs zeroing, forward propagation, loss, backward propagation, communication, and AdamW update. |
| Benchmark cycle | [`run()` — line 412](fsdp2_transformer.py#L412) | Creates the model and optimizer, performs warm-up, times training, reduces metrics, and constructs the result. |
| Program entry point | [`main()` — line 543](fsdp2_transformer.py#L543) | Handles rank-zero reporting, optional JSON output, errors, and process-group cleanup. |

Start with `run()`, then compare the two branches of
`configure_distributed_model()`. The most important FSDP2 detail is that the
optimizer is created only after `fully_shard()` has transformed the model
parameters.

## Run the core comparison

Use the same process count, preset, precision, and batch for both runs. On a
four-GPU node:

```bash
torchrun --standalone --nproc-per-node=4 fsdp2_transformer.py \
    --strategy ddp \
    --preset medium \
    --batch-size-per-device 1 \
    --output result-medium-ddp.json

torchrun --standalone --nproc-per-node=4 fsdp2_transformer.py \
    --strategy fsdp2 \
    --fsdp-wrap block \
    --preset medium \
    --batch-size-per-device 1 \
    --output result-medium-fsdp2-block.json
```

If `medium` is too slow or exceeds memory, substitute `small` in both commands.
Do not shrink only one run: changing the workload invalidates the comparison.

For strong scaling, hold the global batch constant. It must be divisible by the
number of processes:

```bash
torchrun --standalone --nproc-per-node=4 fsdp2_transformer.py \
    --strategy fsdp2 \
    --preset small \
    --global-batch-size 4 \
    --output result-small-global4-fsdp2.json
```

## Run as a single-node Slurm batch job

This layout asks Slurm for one launcher task and four GPUs; `torchrun` creates
one worker process per GPU. Scheduler options differ between clusters, so use
the local GPU-resource spelling when it is not `--gpus-per-node`.

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

set -euo pipefail

srun torchrun --standalone --nproc-per-node=4 fsdp2_transformer.py \
    --strategy ddp \
    --preset medium \
    --output result-medium-ddp.json

srun torchrun --standalone --nproc-per-node=4 fsdp2_transformer.py \
    --strategy fsdp2 \
    --fsdp-wrap block \
    --preset medium \
    --output result-medium-fsdp2-block.json
```

Some sites use `#SBATCH --gres=gpu:4`. This example covers one node only;
multi-node `torchrun` requires rendezvous settings supplied by the batch job.

## Hands-on tasks

Before running anything, predict which strategy will have the lowest peak
memory and which will have the best throughput. Then:

1. Run the DDP and block-wrapped FSDP2 commands above.
2. Compare `local_parameter_fraction`, `peak_allocated_gib`,
   `milliseconds_per_step`, and `tokens_per_second` in the JSON files.
3. Explain why DDP reports a local parameter fraction of approximately `1.0`,
   while FSDP2 approaches `1 / world_size`.
4. Repeat FSDP2 with `--fsdp-wrap root` and predict the result first:

   ```bash
   torchrun --standalone --nproc-per-node=4 fsdp2_transformer.py \
       --strategy fsdp2 \
       --fsdp-wrap root \
       --preset medium \
       --output result-medium-fsdp2-root.json
   ```

5. Explain why root-only and per-block FSDP store similar persistent shards but
   can have different peak memory and throughput.
6. If time and hardware permit, find the largest preset that runs with each
   strategy. Treat an out-of-memory result as capacity evidence, not as a
   throughput measurement.

A successful solution does more than produce three JSON files. It gives a
defensible explanation of the observed trade-off and identifies which strategy
is appropriate when the model fits and when it does not.

## Interpret the output carefully

- `local_parameter_fraction` counts parameter elements stored by the rank. It
  does not include gradients, optimizer state, activations, or communication
  buffers.
- With one process, FSDP2 has nothing to shard and the local fraction remains
  approximately `1.0`; use at least two GPUs to demonstrate its purpose.
- FSDP2 peak memory does not fall exactly by `1 / world_size`: the current
  layer must be gathered for computation, activations remain local, and CUDA
  libraries need temporary memory.
- Block wrapping gives each transformer block its own communication group,
  permitting parameter all-gathers to overlap with computation. Root-only
  wrapping gathers the entire model as one group and is intentionally included
  as a poor boundary choice.
- `parameters_synchronized` is meaningful for replicated DDP parameters. It is
  `null` for FSDP2 because ranks own complementary shards rather than replicas;
  the script instead reports checksums reduced across all shards.
- BF16 is implemented with DDP autocast and an FSDP2 mixed-precision policy.
  The two mechanisms are idiomatic for their respective strategies but do not
  make every intermediate tensor identical.
- Peak memory uses PyTorch's CUDA allocator counters. Allocations made directly
  by libraries such as NCCL are not fully represented.
- Peak statistics are reset after warm-up, so they describe steady-state
  training and deliberately exclude model-construction and initial sharding
  peaks.
- Synthetic, reused input removes I/O noise and model accuracy from the
  exercise. The result must not be presented as end-to-end training
  performance.

Activation checkpointing and distributed checkpoint files are deliberately
left out. Both are valuable follow-up topics, but either would add another
memory mechanism and make the core DDP/FSDP comparison harder to interpret.
