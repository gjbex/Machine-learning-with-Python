# PyTorch activation-checkpointing exercise

[`activation_checkpointing.py`](activation_checkpointing.py) measures the
memory/time trade-off created by activation checkpointing in a native PyTorch
DDP training loop. It uses a synthetic transformer and GPU-resident token
batches, so no model, weights, tokenizer, or dataset are downloaded.

Activation checkpointing retains the inputs to selected transformer blocks but
discards their internal forward activations. Backward recomputes those blocks.
This can reduce peak activation memory, but it adds computation. It does not
shard parameters, gradients, or optimizer state.

## Learning objectives, prerequisites, and duration

After this 40–50 minute exercise, participants should be able to:

1. predict how activation checkpointing changes peak memory and step time;
2. measure the trade-off while keeping model, batch, precision, and GPU count
   fixed;
3. distinguish persistent allocation from the peak created during a training
   cycle;
4. choose checkpoint boundaries based on measured memory pressure rather than
   enabling checkpointing indiscriminately.

Required prerequisites are basic PyTorch training knowledge, the preceding DDP
material, a CUDA-enabled PyTorch environment, and at least one CUDA GPU. More
GPUs can be used, but checkpointing acts independently inside each DDP replica.
The repository's [`environment_gpu.yml`](../../../environment_gpu.yml) supplies
the intended PyTorch environment.

## Workload and safe starting point

Start with the `small` preset and validate it on the actual course hardware.
The `medium` preset makes activation pressure more visible but may require
reducing the local batch on smaller GPUs.

| Preset | Blocks | Hidden size | Sequence | Parameters | FP32 parameters only |
|---|---:|---:|---:|---:|---:|
| `small` | 8 | 512 | 512 | 27,550,720 | 0.10 GiB |
| `medium` | 12 | 768 | 768 | 88,720,896 | 0.33 GiB |

The parameter-memory column excludes gradients, DDP buckets, activations,
temporary attention workspaces, allocator fragmentation, and CUDA/NCCL state.
The script deliberately uses stateless SGD so optimizer state does not obscure
the activation-memory comparison.

`--checkpoint-every 0` disables checkpointing. A value of `1` checkpoints every
transformer block; `2` checkpoints blocks 0, 2, 4, and so forth. Larger
intervals save fewer activations and perform less recomputation.

## Core code map

Function and class names are more durable than line numbers if the file changes.

| Area | Function or class and line | What to inspect |
|---|---|---|
| Options and validation | [`parse_arguments()` — line 90](activation_checkpointing.py#L90) | Defines checkpoint interval, model preset, batch semantics, precision, timing, and protected JSON output. |
| Distributed setup | [`initialize_distributed()` — line 160](activation_checkpointing.py#L160) | Reads `torchrun` variables, binds one GPU per process, and initializes NCCL. |
| Precision setup | [`resolve_precision()` — line 212](activation_checkpointing.py#L212) | Selects BF16 only when every participating GPU supports it. |
| Transformer block | [`TransformerBlock` — line 232](activation_checkpointing.py#L232) | Defines the unit used as a checkpoint/recomputation boundary. |
| Checkpoint selection | [`SyntheticTransformer.forward()` — line 285](activation_checkpointing.py#L285) | Applies non-reentrant `torch.utils.checkpoint.checkpoint()` to the selected blocks. |
| Synthetic input | [`make_batch()` — line 304](activation_checkpointing.py#L304) | Creates a distinct GPU-resident token batch for each rank. |
| Training cycle | [`training_step()` — line 331](activation_checkpointing.py#L331) | Runs forward, backward/recomputation, DDP gradient synchronization, and the optimizer update. |
| Benchmark cycle | [`run()` — line 365](activation_checkpointing.py#L365) | Builds the model, warms up, resets peak statistics, times steps, reduces metrics, and constructs the result. |

Start with `run()`, then inspect `SyntheticTransformer.forward()`. The rest of
the model is intentionally unchanged between checkpointed and uncheckpointed
runs.

## Core comparison

Before running, predict which metric should change most:

- `steady_state_allocated_gib`;
- `peak_allocated_gib`;
- `milliseconds_per_step`.

Use the same GPU count, preset, batch, precision, warm-up, and measured steps in
both runs:

```bash
torchrun --standalone --nproc-per-node=1 activation_checkpointing.py \
    --preset small \
    --checkpoint-every 0 \
    --batch-size-per-device 4 \
    --output result-small-none.json

torchrun --standalone --nproc-per-node=1 activation_checkpointing.py \
    --preset small \
    --checkpoint-every 1 \
    --batch-size-per-device 4 \
    --output result-small-every-block.json
```

Compare the two JSON files and calculate:

```text
memory saving = 1 - checkpointed peak / uncheckpointed peak
time overhead = checkpointed time per step / uncheckpointed time per step - 1
```

There is no universal expected percentage. A successful result keeps the
scientific workload fixed, shows `parameters_synchronized: true`, and supports
an explanation of the observed memory/time trade-off. Normally the checkpointed
run has a lower peak and a longer step time. If the difference is lost in the
persistent-memory baseline, increase the batch or try `medium`—but change the
same option in both comparison runs.

## Hands-on tasks

1. Predict the ordering of steady allocation, peak allocation, and step time
   for checkpoint intervals 0, 1, and 2.
2. Run the uncheckpointed and every-block commands above.
3. Verify that model configuration, batch, precision, and parameter count are
   identical before comparing the peak and time.
4. Run `--checkpoint-every 2`. Decide whether its compromise is preferable on
   the allocated GPU, and justify the decision with both memory and time.
5. Increase `--batch-size-per-device` equally in paired runs. Find one batch
   that fails without checkpointing but succeeds with it, if hardware and time
   permit. Treat the failure as capacity evidence, not as a performance result.
6. Explain why activation checkpointing is a plausible response when memory
   grows with microbatch or sequence length, but not when parameters alone do
   not fit.

The exercise is complete when the participant can recommend one interval—or
recommend no checkpointing—and state the measured reason. Merely producing JSON
files is not sufficient.

## Slurm batch example

This single-node example asks Slurm for one launcher task and two GPUs;
`torchrun` creates one process per GPU. Use local scheduler directives where
they differ.

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00

set -euo pipefail

srun torchrun --standalone --nproc-per-node=2 activation_checkpointing.py \
    --checkpoint-every 0 \
    --batch-size-per-device 4 \
    --output result-none.json

srun torchrun --standalone --nproc-per-node=2 activation_checkpointing.py \
    --checkpoint-every 1 \
    --batch-size-per-device 4 \
    --output result-checkpoint.json
```

For a pure checkpointing comparison, one GPU is enough and avoids DDP
communication noise. Two or more GPUs connect the result to the DDP material
and verify that checkpointing does not replace data parallelism.

## Interpretation and limitations

- `steady_state_allocated_gib` is sampled after warm-up and after gradients are
  cleared. `peak_allocated_gib` captures the maximum live PyTorch allocation
  during measured forward/backward/optimizer cycles.
- Compare allocated memory first. Reserved memory includes cached blocks and may
  remain high after tensors have been released.
- The code uses the recommended non-reentrant checkpoint implementation.
  Checkpoint boundaries are whole transformer blocks, not arbitrary individual
  operations.
- The model has no stochastic dropout, so `preserve_rng_state=False` avoids
  unnecessary random-state work. Real checkpointed regions containing dropout
  normally need RNG-state preservation to reproduce the uncheckpointed
  computation.
- Recomputing every block is not automatically optimal. Partial checkpointing
  can be faster when only a modest amount of memory must be recovered.
- Fused attention kernels already avoid materializing some intermediates.
  Consequently, the saving depends on GPU, PyTorch version, precision, shapes,
  and selected backend.
- CUDA allocator metrics omit some allocations made directly by libraries such
  as NCCL. Synthetic reused inputs also exclude data-pipeline costs and say
  nothing about model accuracy.

See the current PyTorch
[`torch.utils.checkpoint` documentation](https://docs.pytorch.org/docs/stable/checkpoint.html)
for API details and determinism limitations.
