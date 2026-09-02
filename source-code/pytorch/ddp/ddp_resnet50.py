#!/usr/bin/env python3
"""Benchmark ResNet-50 training with native PyTorch DDP.

Synthetic ImageNet-shaped data is created directly on each GPU, so this
measures model computation and DDP communication rather than data loading or
model accuracy. Use ``torchrun`` even for one GPU so every measurement follows
the same code path.

Fixed batch per GPU (weak scaling)::

    torchrun --standalone --nproc-per-node=1 ddp_resnet50.py \
        --batch-size-per-device 32 --output result-1gpu.json
    torchrun --standalone --nproc-per-node=4 ddp_resnet50.py \
        --batch-size-per-device 32 --output result-4gpu.json

Fixed global batch (strong scaling)::

    torchrun --standalone --nproc-per-node=4 ddp_resnet50.py \
        --global-batch-size 128 --output result-4gpu.json

Within a single-node Slurm allocation, set ``--nproc-per-node`` to the number
of GPUs assigned to the job. Multi-node rendezvous settings belong in the
Slurm submission script. The memory figures reported here cover PyTorch's
CUDA allocator; NCCL may allocate additional device memory outside it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50

GIBIBYTE = 2**30


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_arguments() -> argparse.Namespace:
    """Parse arguments and reject unsafe output behavior before GPU startup."""

    parser = argparse.ArgumentParser(
        description="Benchmark synthetic ResNet-50 training with PyTorch DDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    batches = parser.add_mutually_exclusive_group()
    batches.add_argument(
        "--batch-size-per-device",
        type=positive_integer,
        default=32,
        help="samples per GPU; keeping this fixed is a weak-scaling experiment",
    )
    batches.add_argument(
        "--global-batch-size",
        type=positive_integer,
        help="total samples across GPUs; must be divisible by WORLD_SIZE",
    )
    parser.add_argument(
        "--warmup-steps",
        type=positive_integer,
        default=10,
        help="untimed steps used to initialize kernels and DDP buckets",
    )
    parser.add_argument(
        "--steps", type=positive_integer, default=50, help="timed optimizer steps"
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="auto uses BF16 when every GPU supports it, otherwise FP16",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output", type=Path, help="optional JSON result path written by rank zero"
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="allow replacement of an existing output file",
    )
    # Some torchrun versions pass this option; device selection uses LOCAL_RANK.
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()

    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if arguments.output is None and arguments.overwrite_output:
        parser.error("--overwrite-output requires --output")
    if arguments.output is not None:
        if arguments.output.is_dir():
            parser.error(f"output path is a directory: {arguments.output}")
        if arguments.output.exists() and not arguments.overwrite_output:
            parser.error(
                f"output already exists: {arguments.output}; "
                "use --overwrite-output to replace it"
            )
        if not arguments.output.parent.exists():
            parser.error(f"output directory does not exist: {arguments.output.parent}")
    return arguments


def initialize_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize NCCL and bind this process to one visible GPU."""

    required = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"missing torchrun variables {', '.join(missing)}; launch with, for "
            "example, 'torchrun --standalone --nproc-per-node=1 ddp_resnet50.py'"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires GPUs")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    visible_devices = torch.cuda.device_count()
    if not 0 <= local_rank < visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but this process sees "
            f"{visible_devices} CUDA device(s)"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(minutes=5)
    )
    return local_rank, rank, world_size, device


def resolve_batch_sizes(
    arguments: argparse.Namespace, world_size: int
) -> tuple[int, int, str]:
    """Resolve the local and global batches once WORLD_SIZE is known."""

    if arguments.global_batch_size is None:
        per_device = arguments.batch_size_per_device
        return per_device, per_device * world_size, "fixed_per_device"
    if arguments.global_batch_size % world_size != 0:
        raise ValueError(
            f"--global-batch-size={arguments.global_batch_size} is not divisible "
            f"by WORLD_SIZE={world_size}"
        )
    return (
        arguments.global_batch_size // world_size,
        arguments.global_batch_size,
        "fixed_global",
    )


def resolve_precision(
    requested: str, device: torch.device
) -> tuple[str, torch.dtype | None, bool]:
    """Choose one precision supported by every participating GPU."""

    bf16_supported = torch.tensor(
        int(torch.cuda.is_bf16_supported()), dtype=torch.int32, device=device
    )
    dist.all_reduce(bf16_supported, op=dist.ReduceOp.MIN)
    all_support_bf16 = bool(bf16_supported.item())

    if requested == "fp32":
        return "fp32", None, False
    if requested == "bf16":
        if not all_support_bf16:
            raise ValueError(
                "BF16 was requested, but at least one GPU does not support it"
            )
        return "bf16", torch.bfloat16, False
    if requested == "fp16":
        return "fp16", torch.float16, True
    if all_support_bf16:
        return "bf16", torch.bfloat16, False
    return "fp16", torch.float16, True


def make_batch(
    batch_size: int, rank: int, seed: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Create one rank-specific synthetic batch outside the timed region."""

    generator = torch.Generator(device=device).manual_seed(seed + rank)
    images = torch.randn(batch_size, 3, 224, 224, generator=generator, device=device)
    targets = torch.randint(0, 1000, (batch_size,), generator=generator, device=device)
    return images, targets


def training_step(
    model: DDP,
    images: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    autocast_dtype: torch.dtype | None,
) -> Tensor:
    """Run forward, backward, gradient synchronization, and optimizer update."""

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None,
    ):
        loss = nn.functional.cross_entropy(model(images), targets)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss.detach()


def parameter_checksum_difference(model: DDP, device: torch.device) -> float:
    """Return the largest checksum difference between model replicas."""

    checksum = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for parameter in model.module.parameters():
            values = parameter.detach().to(torch.float64)
            checksum[0] += values.sum()
            checksum[1] += values.square().sum()

    minimum = checksum.clone()
    maximum = checksum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return (maximum - minimum).abs().max().item()


def run(arguments: argparse.Namespace) -> dict[str, object]:
    """Run the benchmark and return its result record."""

    local_rank, rank, world_size, device = initialize_distributed()
    per_device_batch, global_batch, batch_mode = resolve_batch_sizes(
        arguments, world_size
    )
    precision, autocast_dtype, scale_gradients = resolve_precision(
        arguments.precision, device
    )

    torch.manual_seed(arguments.seed)
    torch.backends.cudnn.benchmark = True
    model = resnet50(weights=None).to(device).train()
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=scale_gradients)
    images, targets = make_batch(per_device_batch, rank, arguments.seed, device)

    loss = torch.zeros((), device=device)
    for _ in range(arguments.warmup_steps):
        loss = training_step(model, images, targets, optimizer, scaler, autocast_dtype)

    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(arguments.steps):
        loss = training_step(model, images, targets, optimizer, scaler, autocast_dtype)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    # Runtime is limited by the slowest rank; memory reports the largest rank peak.
    maxima = torch.tensor(
        [
            elapsed,
            float(torch.cuda.max_memory_allocated(device)),
            float(torch.cuda.max_memory_reserved(device)),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    elapsed, peak_allocated, peak_reserved = maxima.tolist()

    loss = loss.to(torch.float64)
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    mean_loss = loss.item() / world_size
    if not math.isfinite(mean_loss):
        raise RuntimeError(
            "the final loss is not finite; retry with --precision fp32 or a "
            "smaller batch"
        )
    checksum_difference = parameter_checksum_difference(model, device)

    return {
        "model": "torchvision.models.resnet50",
        "data": "synthetic GPU-resident 3x224x224 images with random labels",
        "batch_mode": batch_mode,
        "batch_size_per_device": per_device_batch,
        "global_batch_size": global_batch,
        "world_size": world_size,
        "warmup_steps": arguments.warmup_steps,
        "measured_steps": arguments.steps,
        "precision_requested": arguments.precision,
        "precision_resolved": precision,
        "seed": arguments.seed,
        "elapsed_seconds": elapsed,
        "milliseconds_per_step": 1000.0 * elapsed / arguments.steps,
        "images_per_second": arguments.steps * global_batch / elapsed,
        "final_loss_mean": mean_loss,
        "peak_allocated_gib": peak_allocated / GIBIBYTE,
        "peak_reserved_gib": peak_reserved / GIBIBYTE,
        "parameter_checksum_max_difference": checksum_difference,
        "parameters_synchronized": checksum_difference == 0.0,
        "gpu_name_rank_zero": torch.cuda.get_device_name(device),
        "hostname_rank_zero": socket.gethostname(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def print_summary(result: dict[str, object]) -> None:
    """Print a compact summary suitable for a Slurm log."""

    print("DDP ResNet-50 synthetic training benchmark")
    print(f"  GPUs:                 {result['world_size']}")
    print(f"  precision:            {result['precision_resolved']}")
    print(f"  batch/device:         {result['batch_size_per_device']}")
    print(f"  global batch:         {result['global_batch_size']}")
    print(f"  time/step:            {result['milliseconds_per_step']:.3f} ms")
    print(f"  throughput:           {result['images_per_second']:.1f} images/s")
    print(f"  peak allocated/GPU:   {result['peak_allocated_gib']:.2f} GiB")
    print(f"  peak reserved/GPU:    {result['peak_reserved_gib']:.2f} GiB")
    print(f"  parameters synchronized: {result['parameters_synchronized']}")


def main() -> int:
    """Parse options, run the benchmark, and write rank-zero output."""

    arguments = parse_arguments()
    try:
        result = run(arguments)
        if dist.get_rank() == 0:
            print_summary(result)
            if arguments.output is not None:
                text = json.dumps(result, indent=2, sort_keys=True)
                arguments.output.write_text(f"{text}\n", encoding="utf-8")
                print(f"  JSON result:          {arguments.output}")
        return 0
    except (RuntimeError, ValueError) as error:
        rank = os.environ.get("RANK", "unknown")
        print(f"rank {rank}: error: {error}", file=sys.stderr)
        return 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
