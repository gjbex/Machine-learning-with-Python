#!/usr/bin/env python3
"""Benchmark ResNet-50 training with PyTorch Lightning DDP.

This is the Lightning counterpart of
``source-code/pytorch/ddp/ddp_resnet50.py``. It uses the same model, synthetic
ImageNet-shaped inputs, optimizer, batch modes, warm-up period, measurements,
and JSON result fields. Lightning owns process and device setup, DDP wrapping,
mixed precision, gradient synchronization, and optimizer execution.

Inside a single-node Slurm allocation with four tasks and four GPUs::

    srun python ddp_resnet50.py --devices 4 \
        --batch-size-per-device 32 --output result-4gpu.json

Outside Slurm, Lightning can launch the processes itself::

    python ddp_resnet50.py --devices 4 \
        --batch-size-per-device 32 --output result-4gpu.json

The benchmark uses Lightning's barebones Trainer mode. The memory figures
cover PyTorch's CUDA allocator; NCCL may allocate additional device memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from datetime import timedelta
from pathlib import Path

import lightning as L
import torch
import torchvision
from lightning.pytorch.strategies import DDPStrategy
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader
from torchvision.models import resnet50

GIBIBYTE = 2**30


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_arguments() -> argparse.Namespace:
    """Parse arguments and reject invalid runs before Trainer startup."""

    parser = argparse.ArgumentParser(
        description="Benchmark synthetic ResNet-50 training with Lightning DDP.",
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
        help="total samples across GPUs; must be divisible by the world size",
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
        help="auto uses BF16 when supported, otherwise FP16",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--devices",
        type=positive_integer,
        default=1,
        help="GPUs per node; under Slurm this must match --ntasks-per-node",
    )
    parser.add_argument(
        "--num-nodes", type=positive_integer, default=1, help="Slurm node count"
    )
    parser.add_argument(
        "--output", type=Path, help="optional JSON result path written by rank zero"
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="allow replacement of an existing output file",
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

    expected_world_size = arguments.devices * arguments.num_nodes
    if (
        arguments.global_batch_size is not None
        and arguments.global_batch_size % expected_world_size != 0
    ):
        parser.error(
            f"--global-batch-size={arguments.global_batch_size} is not divisible "
            f"by devices x nodes={expected_world_size}"
        )
    return arguments


def resolve_trainer_precision(requested: str) -> tuple[str, str]:
    """Map the shared CLI vocabulary to Lightning's Trainer vocabulary."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires GPUs")
    if requested == "fp32":
        return "32-true", "fp32"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "BF16 was requested, but the selected GPU does not support it"
            )
        return "bf16-mixed", "bf16"
    if requested == "fp16":
        return "16-mixed", "fp16"
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed", "bf16"
    return "16-mixed", "fp16"


class ResNet50Benchmark(L.LightningModule):
    """ResNet-50 workload plus benchmark-specific timing and reporting hooks."""

    def __init__(
        self,
        *,
        batch_size_per_device: int,
        global_batch_size: int | None,
        warmup_steps: int,
        measured_steps: int,
        precision_requested: str,
        precision_resolved: str,
        seed: int,
        output: Path | None,
    ) -> None:
        super().__init__()
        self.model = resnet50(weights=None)
        self.batch_size_per_device_requested = batch_size_per_device
        self.global_batch_size_requested = global_batch_size
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.precision_requested = precision_requested
        self.precision_resolved = precision_resolved
        self.seed = seed
        self.output = output

        self._images: Tensor | None = None
        self._targets: Tensor | None = None
        self._last_loss: Tensor | None = None
        self._start_time: float | None = None
        self._maxima: Tensor | None = None
        self._batch_mode = ""
        self._batch_size_per_device = 0
        self._global_batch_size = 0

    def forward(self, images: Tensor) -> Tensor:
        """Apply ResNet-50 to a batch of images."""

        return self.model(images)

    def setup(self, stage: str) -> None:
        """Resolve batch semantics after Lightning establishes the world size."""

        if stage != "fit":
            return
        world_size = self.trainer.world_size
        if self.global_batch_size_requested is None:
            self._batch_size_per_device = self.batch_size_per_device_requested
            self._global_batch_size = self._batch_size_per_device * world_size
            self._batch_mode = "fixed_per_device"
        else:
            if self.global_batch_size_requested % world_size != 0:
                raise ValueError(
                    f"--global-batch-size={self.global_batch_size_requested} is not "
                    f"divisible by the actual world size {world_size}"
                )
            self._batch_size_per_device = self.global_batch_size_requested // world_size
            self._global_batch_size = self.global_batch_size_requested
            self._batch_mode = "fixed_global"

    def on_train_start(self) -> None:
        """Create one rank-specific synthetic batch on Lightning's chosen device."""

        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + self.global_rank
        )
        self._images = torch.randn(
            self._batch_size_per_device,
            3,
            224,
            224,
            generator=generator,
            device=self.device,
        )
        self._targets = torch.randint(
            0,
            1000,
            (self._batch_size_per_device,),
            generator=generator,
            device=self.device,
        )

    def training_step(self, _batch: int, _batch_idx: int) -> Tensor:
        """Compute the loss; Lightning runs backward and the optimizer update."""

        if self._images is None or self._targets is None:
            raise RuntimeError("synthetic inputs were not initialized")
        loss = nn.functional.cross_entropy(self(self._images), self._targets)
        self._last_loss = loss.detach()
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use the same optimizer as the native DDP benchmark."""

        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def train_dataloader(self) -> DataLoader:
        """Supply one lightweight token for every optimizer step."""

        total_steps = self.warmup_steps + self.measured_steps
        return DataLoader(range(total_steps), batch_size=None, num_workers=0)

    def on_train_batch_start(self, _batch: int, batch_idx: int) -> None:
        """Start synchronized timing after the warm-up batches."""

        if batch_idx != self.warmup_steps:
            return
        self.trainer.strategy.barrier()
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._start_time = time.perf_counter()

    def on_train_batch_end(self, _outputs: Tensor, _batch: int, batch_idx: int) -> None:
        """Stop timing after the final measured optimizer step."""

        total_steps = self.warmup_steps + self.measured_steps
        if batch_idx + 1 != total_steps:
            return
        if self._start_time is None:
            raise RuntimeError("benchmark timer was not started")

        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self._start_time
        local = torch.tensor(
            [
                elapsed,
                float(torch.cuda.max_memory_allocated(self.device)),
                float(torch.cuda.max_memory_reserved(self.device)),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        self._maxima = self.trainer.strategy.reduce(local, reduce_op=ReduceOp.MAX)

    def _parameter_checksum_difference(self) -> float:
        """Return the largest checksum difference between model replicas."""

        checksum = torch.zeros(2, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for parameter in self.parameters():
                values = parameter.detach().to(torch.float64)
                checksum[0] += values.sum()
                checksum[1] += values.square().sum()
        minimum = self.trainer.strategy.reduce(checksum.clone(), reduce_op=ReduceOp.MIN)
        maximum = self.trainer.strategy.reduce(checksum, reduce_op=ReduceOp.MAX)
        difference = maximum - minimum
        return difference.abs().max().item()

    def on_train_end(self) -> None:
        """Reduce correctness metrics and write one result on global rank zero."""

        if self._last_loss is None or self._maxima is None:
            raise RuntimeError("the benchmark did not complete all requested steps")

        mean_loss = self.trainer.strategy.reduce(
            self._last_loss.to(torch.float64), reduce_op="mean"
        ).item()
        if not math.isfinite(mean_loss):
            raise RuntimeError(
                "the final loss is not finite; retry with --precision fp32 or a "
                "smaller batch"
            )
        checksum_difference = self._parameter_checksum_difference()
        elapsed, peak_allocated, peak_reserved = self._maxima.tolist()

        result = {
            "model": "torchvision.models.resnet50",
            "data": "synthetic GPU-resident 3x224x224 images with random labels",
            "batch_mode": self._batch_mode,
            "batch_size_per_device": self._batch_size_per_device,
            "global_batch_size": self._global_batch_size,
            "world_size": self.trainer.world_size,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "precision_requested": self.precision_requested,
            "precision_resolved": self.precision_resolved,
            "seed": self.seed,
            "elapsed_seconds": elapsed,
            "milliseconds_per_step": 1000.0 * elapsed / self.measured_steps,
            "images_per_second": (
                self.measured_steps * self._global_batch_size / elapsed
            ),
            "final_loss_mean": mean_loss,
            "peak_allocated_gib": peak_allocated / GIBIBYTE,
            "peak_reserved_gib": peak_reserved / GIBIBYTE,
            "parameter_checksum_max_difference": checksum_difference,
            "parameters_synchronized": checksum_difference == 0.0,
            "gpu_name_rank_zero": torch.cuda.get_device_name(self.device),
            "hostname_rank_zero": socket.gethostname(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "lightning_version": L.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "lightning_barebones": True,
        }
        if self.trainer.is_global_zero:
            print_summary(result)
            if self.output is not None:
                text = json.dumps(result, indent=2, sort_keys=True)
                self.output.write_text(f"{text}\n", encoding="utf-8")
                print(f"  JSON result:          {self.output}")


def print_summary(result: dict[str, object]) -> None:
    """Print a compact summary suitable for a Slurm log."""

    print("Lightning DDP ResNet-50 synthetic training benchmark")
    print(f"  GPUs:                 {result['world_size']}")
    print(f"  precision:            {result['precision_resolved']}")
    print(f"  batch/device:         {result['batch_size_per_device']}")
    print(f"  global batch:         {result['global_batch_size']}")
    print(f"  time/step:            {result['milliseconds_per_step']:.3f} ms")
    print(f"  throughput:           {result['images_per_second']:.1f} images/s")
    print(f"  peak allocated/GPU:   {result['peak_allocated_gib']:.2f} GiB")
    print(f"  peak reserved/GPU:    {result['peak_reserved_gib']:.2f} GiB")
    print(f"  parameters synchronized: {result['parameters_synchronized']}")


def main() -> None:
    """Create the Lightning module and run the barebones DDP Trainer."""

    arguments = parse_arguments()
    trainer_precision, resolved_precision = resolve_trainer_precision(
        arguments.precision
    )
    L.seed_everything(arguments.seed, workers=True)
    model = ResNet50Benchmark(
        batch_size_per_device=arguments.batch_size_per_device,
        global_batch_size=arguments.global_batch_size,
        warmup_steps=arguments.warmup_steps,
        measured_steps=arguments.steps,
        precision_requested=arguments.precision,
        precision_resolved=resolved_precision,
        seed=arguments.seed,
        output=arguments.output,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=arguments.devices,
        num_nodes=arguments.num_nodes,
        strategy=DDPStrategy(
            timeout=timedelta(minutes=5), gradient_as_bucket_view=True
        ),
        precision=trainer_precision,
        max_steps=arguments.warmup_steps + arguments.steps,
        max_epochs=1,
        use_distributed_sampler=False,
        barebones=True,
        benchmark=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
