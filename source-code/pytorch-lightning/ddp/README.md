# PyTorch Lightning DDP benchmark

[`ddp_resnet50.py`](ddp_resnet50.py) is the PyTorch Lightning counterpart of
the [native PyTorch DDP benchmark](../../pytorch/ddp/ddp_resnet50.py). It uses
the same randomly initialized ResNet-50, synthetic ImageNet-shaped inputs,
optimizer, batch modes, warm-up period, timed steps, and result metrics.

No model weights or dataset are downloaded. The ResNet-50 implementation is
part of `torchvision`, and `weights=None` requests a random initialization.

The example is intended to compare one and multiple GPUs and, alongside the
native version, to show which DDP responsibilities Lightning takes over. It
measures model computation, Lightning's training-loop overhead, and gradient
communication. It does not measure data-loading performance or model accuracy.

## Requirements

The script requires CUDA-enabled PyTorch, `torchvision`, and Lightning 2.x
using the current `lightning` package and import name. Lightning is an optional
dependency and is not added to the repository's core GPU environment:

```bash
python -m pip install lightning
```

On a cluster, prefer installing it in a project-specific environment rather
than altering a shared system Python installation.

## Core code map

The line numbers below refer to the current version of
[`ddp_resnet50.py`](ddp_resnet50.py). Function and method names are the more
durable reference if the file changes later.

| Area | Function or method and line | What to inspect |
|---|---|---|
| Input and option validation | [`parse_arguments()` — line 57](ddp_resnet50.py#L57) | Defines the batch, timing, precision, device, node, and output options and rejects inconsistent runs. |
| Precision setup | [`resolve_trainer_precision()` — line 138](ddp_resnet50.py#L138) | Maps the shared FP32/BF16/FP16 options to Lightning's precision modes. |
| Model creation | [`ResNet50Benchmark.__init__()` — line 161](ddp_resnet50.py#L161) | Creates `torchvision.models.resnet50(weights=None)` at line 174 and stores the benchmark configuration. |
| Distributed batch interpretation | [`ResNet50Benchmark.setup()` — line 198](ddp_resnet50.py#L198) | Uses Lightning's actual world size to distinguish a fixed per-device batch from a fixed global batch. |
| Synthetic input | [`ResNet50Benchmark.on_train_start()` — line 218](ddp_resnet50.py#L218) | Creates a distinct GPU-resident image batch and labels for each rank. |
| One training cycle | [`ResNet50Benchmark.training_step()` — line 240](ddp_resnet50.py#L240) | Runs the forward pass and loss calculation. Lightning performs backward propagation, DDP gradient synchronization, and the optimizer update. |
| Optimizer setup | [`ResNet50Benchmark.configure_optimizers()` — line 249](ddp_resnet50.py#L249) | Creates the same SGD optimizer as the native example. |
| Training-loop input | [`ResNet50Benchmark.train_dataloader()` — line 254](ddp_resnet50.py#L254) | Supplies one lightweight token per optimizer step; the real synthetic images already reside on each GPU. |
| Warm-up and measured cycles | [`on_train_batch_start()` — line 260](ddp_resnet50.py#L260) and [`on_train_batch_end()` — line 270](ddp_resnet50.py#L270) | Synchronize the ranks, exclude warm-up steps, time the measured steps, and collect peak CUDA memory. |
| DDP and Trainer setup | [`main()` — line 379](ddp_resnet50.py#L379) | Creates the module and configures Lightning's GPU accelerator, `DDPStrategy`, precision plugin, and barebones Trainer. |

Start with `main()` to see the high-level Lightning setup, then compare
`training_step()` with the native benchmark's `training_step()`.

## What Lightning changes

| Responsibility | Native PyTorch example | Lightning example |
|---|---|---|
| Process and NCCL setup | Explicit process-group initialization and teardown | `Trainer` and `DDPStrategy` |
| Device placement and DDP wrapper | Explicit local-rank device selection and `DistributedDataParallel(...)` | Lightning accelerator and strategy |
| Mixed precision | Explicit autocast and gradient scaler | Trainer `precision` setting |
| Backward pass and optimizer update | Written in the training step | Automatic after `training_step()` returns the loss |
| Benchmark timing and reporting | Custom code | Still custom code, because these are benchmark requirements rather than ordinary training features |

This is intentionally an exact comparison rather than the shortest possible
Lightning program. Consequently, validation, timing, JSON reporting, and
correctness checks still account for much of the file. The simplification is
most visible in `main()` and `training_step()`.

## Run as a Slurm batch job

Lightning detects the Slurm environment from its variables. The Slurm task
count per node, allocated GPU count, and `--devices` value must agree. For a
single-node, four-GPU job, the relevant batch-script lines are typically:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

srun python ddp_resnet50.py \
    --devices 4 \
    --batch-size-per-device 32 \
    --output result-4gpu.json
```

Some clusters spell the GPU request as `#SBATCH --gres=gpu:4`; use the syntax
documented by the local site. For the one-GPU baseline, change all three `4`
values to `1` and use a different output name.

For weak scaling, keep `--batch-size-per-device` fixed while changing the GPU
count. For strong scaling, keep the global batch fixed:

```bash
srun python ddp_resnet50.py \
    --devices 4 \
    --global-batch-size 128 \
    --output result-global128-4gpu.json
```

Outside Slurm, Lightning can launch the DDP worker processes itself:

```bash
python ddp_resnet50.py \
    --devices 4 \
    --batch-size-per-device 32 \
    --output result-4gpu.json
```

## Interpreting the result

The JSON result records throughput, time per optimizer step, peak PyTorch CUDA
memory, the resolved precision, software versions, hardware information, and a
check that model parameters remained synchronized.

There are several deliberate limitations:

- `barebones=True` removes logging, checkpointing, progress bars, and other
  Trainer services so that they do not distort a short benchmark. It is not a
  recommended default for a real training application.
- The synthetic batch lives on the GPU and is reused. This isolates model and
  communication performance but says nothing about storage or data-pipeline
  scaling.
- The timed region includes Lightning's per-step loop overhead. That is useful
  for this comparison, but means a small difference from the native result is
  not necessarily a DDP communication difference.
- `--precision auto` assumes homogeneous GPUs across all nodes. Select an
  explicit precision on a heterogeneous allocation.
- PyTorch's CUDA memory counters do not include every allocation made directly
  by CUDA libraries such as NCCL.
