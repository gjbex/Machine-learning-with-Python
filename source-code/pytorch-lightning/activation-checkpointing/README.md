# PyTorch Lightning activation-checkpointing exercise

[`activation_checkpointing.py`](activation_checkpointing.py) is the Lightning
counterpart of the
[native PyTorch exercise](../../pytorch/activation-checkpointing/). It uses the
same synthetic transformer, presets, checkpoint boundaries, optimizer, batch
semantics, warm-up, timed steps, and output metrics.

This comparison exposes an important abstraction boundary: Lightning owns the
distributed training machinery, but activation checkpointing remains a model
decision. The model still calls PyTorch's `torch.utils.checkpoint`; this is not
the same feature as Lightning's `ModelCheckpoint`, which saves training state
to storage.

## Requirements and duration

As a follow-on to the native exercise, allow 25–35 minutes. Used independently,
allow 40–50 minutes. The requirements are CUDA-enabled PyTorch, Lightning 2.x,
and at least one CUDA GPU. Lightning is an optional dependency and is not part
of the repository's core GPU environment:

```bash
python -m pip install lightning
```

Install it before the session in a project-specific environment or course
container rather than modifying a shared cluster Python installation.

## Core code map

Function and class names are more durable than line numbers if the file changes.

| Area | Function or class and line | What to inspect |
|---|---|---|
| Options and validation | [`parse_arguments()` — line 91](activation_checkpointing.py#L91) | Defines the checkpoint interval, preset, batch, timing, devices, nodes, precision, and protected output. |
| Precision setup | [`resolve_trainer_precision()` — line 177](activation_checkpointing.py#L177) | Maps FP32/BF16 choices to Lightning Trainer precision modes. |
| Transformer block | [`TransformerBlock` — line 195](activation_checkpointing.py#L195) | Defines the model unit selected for recomputation. |
| Checkpoint selection | [`SyntheticTransformer.forward()` — line 248](activation_checkpointing.py#L248) | Uses the same non-reentrant PyTorch checkpoint call as the native implementation. |
| Lightning module | [`ActivationCheckpointBenchmark` — line 267](activation_checkpointing.py#L267) | Owns the synthetic workload plus benchmark-specific measurement hooks. |
| Batch interpretation | [`setup()` — line 313](activation_checkpointing.py#L313) | Resolves local/global batch semantics after Lightning establishes the world size. |
| Synthetic input | [`on_train_start()` — line 333](activation_checkpointing.py#L333) | Creates a distinct GPU-resident token batch on each worker. |
| Training cycle | [`training_step()` — line 354](activation_checkpointing.py#L354) | Returns the forward loss; Lightning performs backward, recomputation, synchronization, and optimizer update. |
| Measurement | [`on_train_batch_start()` — line 375](activation_checkpointing.py#L375) and [`on_train_batch_end()` — line 393](activation_checkpointing.py#L393) | Exclude warm-up, synchronize workers, time steps, and record peak CUDA memory. |
| Result construction | [`on_train_end()` — line 423](activation_checkpointing.py#L423) | Reduces correctness/performance metrics and writes the rank-zero JSON result. |
| Trainer setup | [`main()` — line 513](activation_checkpointing.py#L513) | Configures devices, nodes, DDP, precision, and a barebones training loop. |

## Run the matched comparison

Outside Slurm, Lightning can launch the workers itself:

```bash
python activation_checkpointing.py \
    --devices 1 \
    --preset small \
    --checkpoint-every 0 \
    --batch-size-per-device 4 \
    --output result-small-none.json

python activation_checkpointing.py \
    --devices 1 \
    --preset small \
    --checkpoint-every 1 \
    --batch-size-per-device 4 \
    --output result-small-every-block.json
```

Under Slurm, Lightning expects one task per GPU. The task count, allocated GPU
count, and `--devices` value must agree:

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00

set -euo pipefail

srun python activation_checkpointing.py \
    --devices 2 \
    --checkpoint-every 0 \
    --output result-none.json

srun python activation_checkpointing.py \
    --devices 2 \
    --checkpoint-every 1 \
    --output result-checkpoint.json
```

Cluster launch conventions vary. Validate this pattern with the local Slurm and
Lightning configuration before the course.

## Follow-on tasks

1. Predict which parts of the native script Lightning should remove and which
   parts must remain because they define the scientific model or measurement.
2. Run the matched Lightning comparison and verify the configuration fields
   before comparing `peak_allocated_gib` and `milliseconds_per_step`.
3. Compare `SyntheticTransformer.forward()` in both implementations. Explain
   why the checkpoint boundaries remain explicit even though backward and the
   optimizer step disappeared from Lightning's `training_step()`.
4. Compare native and Lightning results cautiously. Small timing differences
   can be framework-loop overhead rather than checkpoint recomputation.
5. Decide whether the abstraction saves meaningful research-code complexity in
   this example. Support the answer with concrete responsibilities rather than
   line count alone.

The checkpointing conclusion should agree across implementations: selected
forward activations are exchanged for backward recomputation. Exact memory and
timing numbers need not be identical.

## Important limitations

- `barebones=True` removes loggers, progress bars, storage checkpoints, and
  other Trainer services that would distort a short benchmark. It is not a
  normal production Trainer configuration.
- `--precision auto` is resolved before Lightning launches distributed workers
  and assumes homogeneous GPUs. Use an explicit precision on heterogeneous
  nodes.
- The model contains no dropout, so the checkpoint call disables RNG-state
  preservation. Preserve random state when checkpointed production blocks use
  stochastic operations.
- Lightning's `ModelCheckpoint` and activation checkpointing solve unrelated
  problems: recoverability versus activation memory.
- The workload is synthetic, reuses one GPU-resident batch, and reports
  PyTorch-allocator memory. It is not an end-to-end training benchmark.

For the full task sequence, model presets, interpretation guidance, and
checkpoint-interval experiment, follow the
[native exercise README](../../pytorch/activation-checkpointing/README.md).
