# PyTorch DistributedDataParallel benchmark

[`ddp_resnet50.py`](ddp_resnet50.py) measures the training throughput of a
randomly initialized ResNet-50 using native PyTorch
`DistributedDataParallel` (DDP). It creates synthetic ImageNet-shaped inputs
directly on each GPU, so no model weights or dataset are downloaded.

The example is intended to compare one and multiple GPUs. It measures model
computation and gradient communication, not data-loading performance or model
accuracy.

## Core code map

The line numbers below refer to the current version of
[`ddp_resnet50.py`](ddp_resnet50.py). Function names are the more durable
reference if the file changes later.

| Area | Function and line | What to inspect |
|---|---|---|
| Distributed setup | [`initialize_distributed()` — line 125](ddp_resnet50.py#L125) | Reads the environment created by `torchrun`, selects one GPU per process, and initializes the NCCL process group. |
| Batch interpretation | [`resolve_batch_sizes()` — line 156](ddp_resnet50.py#L156) | Distinguishes a fixed per-device batch from a fixed global batch. |
| Precision setup | [`resolve_precision()` — line 176](ddp_resnet50.py#L176) | Selects FP32, BF16, or FP16 consistently across all ranks. |
| Synthetic input | [`make_batch()` — line 202](ddp_resnet50.py#L202) | Creates a distinct GPU-resident image batch and labels for each rank. |
| One training cycle | [`training_step()` — line 213](ddp_resnet50.py#L213) | Runs zeroing, forward propagation, loss calculation, backward propagation, DDP gradient synchronization, and the optimizer update. |
| Model creation and DDP wrapper | [`run()` — line 258](ddp_resnet50.py#L258) | Creates ResNet-50 at line 271, wraps it in DDP at line 272, and creates the optimizer and gradient scaler at lines 278–279. |
| Warm-up and measured cycles | [`run()` — line 258](ddp_resnet50.py#L258) | Executes warm-up steps at lines 283–284 and the timed training loop at lines 286–305. |

The `run()` function is the best starting point for reading the program because
it shows how the setup, input, model, and training-cycle functions fit together.

## Run inside a Slurm allocation

Use `torchrun` for both the one-GPU baseline and multi-GPU measurements so that
all runs follow the same DDP code path. For example, on a node assigned four
GPUs:

```bash
torchrun --standalone --nproc-per-node=4 ddp_resnet50.py \
    --batch-size-per-device 32 \
    --output result-4gpu.json
```

For weak scaling, keep `--batch-size-per-device` fixed while changing the GPU
count. For strong scaling, keep the global batch fixed:

```bash
torchrun --standalone --nproc-per-node=4 ddp_resnet50.py \
    --global-batch-size 128 \
    --output result-global128-4gpu.json
```

The JSON output records throughput, time per optimizer step, peak PyTorch CUDA
memory, the resolved precision, software versions, hardware information, and a
check that model parameters remained synchronized. PyTorch's memory counters do
not include every allocation made directly by CUDA libraries such as NCCL.
