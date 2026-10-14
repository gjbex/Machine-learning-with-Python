#!/usr/bin/env python3
"""Measure activation-checkpoint memory and time with Lightning DDP.

This is the PyTorch Lightning counterpart of the native PyTorch activation-
checkpointing exercise. Compare otherwise identical runs with
``--checkpoint-every 0`` and ``--checkpoint-every 1``::

    python activation_checkpointing.py --devices 1 \
        --checkpoint-every 0 --output result-no-checkpoint.json
    python activation_checkpointing.py --devices 1 \
        --checkpoint-every 1 --output result-checkpoint.json

Lightning owns the distributed training loop, but activation checkpoint
boundaries remain part of the model and use ``torch.utils.checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.strategies import DDPStrategy
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

GIBIBYTE = 2**30


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one synthetic transformer preset."""

    layers: int
    hidden_size: int
    attention_heads: int
    feedforward_size: int
    sequence_length: int
    vocabulary_size: int = 4096
    classes: int = 16


PRESETS = {
    "small": ModelConfig(
        layers=8,
        hidden_size=512,
        attention_heads=8,
        feedforward_size=2048,
        sequence_length=512,
    ),
    "medium": ModelConfig(
        layers=12,
        hidden_size=768,
        attention_heads=12,
        feedforward_size=3072,
        sequence_length=768,
    ),
}


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def non_negative_integer(value: str) -> int:
    """Parse a non-negative command-line integer."""

    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def parse_arguments() -> argparse.Namespace:
    """Parse arguments and reject invalid runs before Trainer startup."""

    parser = argparse.ArgumentParser(
        description="Benchmark activation checkpointing with Lightning DDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument(
        "--checkpoint-every",
        type=non_negative_integer,
        default=0,
        metavar="N",
        help="checkpoint every Nth transformer block; 0 disables checkpointing",
    )
    batches = parser.add_mutually_exclusive_group()
    batches.add_argument(
        "--batch-size-per-device",
        type=positive_integer,
        default=4,
        help="samples per GPU; keeping this fixed is weak scaling",
    )
    batches.add_argument(
        "--global-batch-size",
        type=positive_integer,
        help="total samples across GPUs; must be divisible by the world size",
    )
    parser.add_argument(
        "--warmup-steps", type=positive_integer, default=3, help="untimed steps"
    )
    parser.add_argument(
        "--steps", type=positive_integer, default=10, help="timed optimizer steps"
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="auto uses BF16 when supported, otherwise FP32",
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
    """Map the shared CLI vocabulary to Lightning's precision vocabulary."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires a GPU")
    if requested == "fp32":
        return "32-true", "fp32"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "BF16 was requested, but the selected GPU does not support it"
            )
        return "bf16-mixed", "bf16"
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed", "bf16"
    return "32-true", "fp32"


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention and feed-forward transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.attention_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(config.hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(config.hidden_size, config.feedforward_size, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.feedforward_size, config.hidden_size, bias=False),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply one residual attention and feed-forward block."""

        normalized = self.attention_norm(hidden_states)
        attention_output, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden_states = hidden_states + attention_output
        return hidden_states + self.feedforward(self.feedforward_norm(hidden_states))


class SyntheticTransformer(nn.Module):
    """Transformer whose selected blocks discard forward activations."""

    def __init__(self, config: ModelConfig, checkpoint_every: int) -> None:
        super().__init__()
        self.checkpoint_every = checkpoint_every
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.empty(config.sequence_length, config.hidden_size)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.layers)
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.classes, bias=False)

    def should_checkpoint(self, block_index: int) -> bool:
        """Return whether a block belongs to the selected checkpoint interval."""

        return self.checkpoint_every > 0 and block_index % self.checkpoint_every == 0

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return logits while checkpointing the selected transformer blocks."""

        hidden_states = self.token_embedding(token_ids)
        hidden_states = hidden_states + self.position_embedding.unsqueeze(0)
        for block_index, block in enumerate(self.blocks):
            if self.should_checkpoint(block_index):
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden_states = block(hidden_states)
        hidden_states = self.output_norm(hidden_states[:, 0])
        return self.classifier(hidden_states)


class ActivationCheckpointBenchmark(L.LightningModule):
    """Synthetic transformer plus Lightning timing and reporting hooks."""

    def __init__(
        self,
        *,
        config: ModelConfig,
        preset: str,
        checkpoint_every: int,
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
        self.model = SyntheticTransformer(config, checkpoint_every)
        self.config = config
        self.preset = preset
        self.checkpoint_every = checkpoint_every
        self.batch_size_per_device_requested = batch_size_per_device
        self.global_batch_size_requested = global_batch_size
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.precision_requested = precision_requested
        self.precision_resolved = precision_resolved
        self.seed = seed
        self.output = output

        self._token_ids: Tensor | None = None
        self._targets: Tensor | None = None
        self._last_loss: Tensor | None = None
        self._start_time: float | None = None
        self._maxima: Tensor | None = None
        self._batch_mode = ""
        self._batch_size_per_device = 0
        self._global_batch_size = 0

    def forward(self, token_ids: Tensor) -> Tensor:
        """Apply the synthetic transformer."""

        return self.model(token_ids)

    def setup(self, stage: str) -> None:
        """Resolve batch semantics after Lightning establishes world size."""

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
                    f"--global-batch-size={self.global_batch_size_requested} is "
                    f"not divisible by the actual world size {world_size}"
                )
            self._batch_size_per_device = self.global_batch_size_requested // world_size
            self._global_batch_size = self.global_batch_size_requested
            self._batch_mode = "fixed_global"

    def on_train_start(self) -> None:
        """Create one rank-specific synthetic batch on the selected GPU."""

        generator = torch.Generator(device=self.device).manual_seed(
            self.seed + self.global_rank
        )
        self._token_ids = torch.randint(
            0,
            self.config.vocabulary_size,
            (self._batch_size_per_device, self.config.sequence_length),
            generator=generator,
            device=self.device,
        )
        self._targets = torch.randint(
            0,
            self.config.classes,
            (self._batch_size_per_device,),
            generator=generator,
            device=self.device,
        )

    def training_step(self, _batch: int, _batch_idx: int) -> Tensor:
        """Compute loss; Lightning runs backward and the optimizer update."""

        if self._token_ids is None or self._targets is None:
            raise RuntimeError("synthetic inputs were not initialized")
        logits = self(self._token_ids)
        loss = nn.functional.cross_entropy(logits.float(), self._targets)
        self._last_loss = loss.detach()
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use the same stateless optimizer as the native benchmark."""

        return torch.optim.SGD(self.parameters(), lr=1e-3)

    def train_dataloader(self) -> DataLoader:
        """Supply one lightweight token for every optimizer step."""

        total_steps = self.warmup_steps + self.measured_steps
        return DataLoader(range(total_steps), batch_size=None, num_workers=0)

    def on_train_batch_start(self, _batch: int, batch_idx: int) -> None:
        """Start synchronized measurement after the warm-up batches."""

        if batch_idx != self.warmup_steps:
            return
        for optimizer in self.trainer.optimizers:
            optimizer.zero_grad(set_to_none=True)
        self.trainer.strategy.barrier()
        torch.cuda.synchronize(self.device)
        steady_state_allocated = float(torch.cuda.memory_allocated(self.device))
        torch.cuda.reset_peak_memory_stats(self.device)
        self._start_time = time.perf_counter()
        self._maxima = torch.tensor(
            [0.0, steady_state_allocated, 0.0, 0.0],
            dtype=torch.float64,
            device=self.device,
        )

    def on_train_batch_end(self, _outputs: Tensor, _batch: int, batch_idx: int) -> None:
        """Stop timing after the final measured optimizer step."""

        total_steps = self.warmup_steps + self.measured_steps
        if batch_idx + 1 != total_steps:
            return
        if self._start_time is None or self._maxima is None:
            raise RuntimeError("benchmark timer was not started")

        torch.cuda.synchronize(self.device)
        self._maxima[0] = time.perf_counter() - self._start_time
        self._maxima[2] = float(torch.cuda.max_memory_allocated(self.device))
        self._maxima[3] = float(torch.cuda.max_memory_reserved(self.device))
        self._maxima = self.trainer.strategy.reduce(
            self._maxima, reduce_op=ReduceOp.MAX
        )

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
        return (maximum - minimum).abs().max().item()

    def on_train_end(self) -> None:
        """Reduce correctness metrics and write one rank-zero result."""

        if self._last_loss is None or self._maxima is None:
            raise RuntimeError("the benchmark did not complete all requested steps")

        mean_loss = self.trainer.strategy.reduce(
            self._last_loss.to(torch.float64), reduce_op="mean"
        ).item()
        if not math.isfinite(mean_loss):
            raise RuntimeError(
                "the final loss is not finite; retry with --precision fp32 or "
                "the small preset"
            )
        checksum_difference = self._parameter_checksum_difference()
        elapsed, steady_state_allocated, peak_allocated, peak_reserved = (
            self._maxima.tolist()
        )
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        checkpointed_blocks = sum(
            self.model.should_checkpoint(index) for index in range(self.config.layers)
        )
        samples = self.measured_steps * self._global_batch_size
        tokens = samples * self.config.sequence_length

        result = {
            "model": "synthetic encoder-style transformer",
            "data": "synthetic GPU-resident integer token sequences",
            "preset": self.preset,
            "model_config": asdict(self.config),
            "parameter_count": parameter_count,
            "model_parameters_fp32_gib": 4 * parameter_count / GIBIBYTE,
            "checkpoint_every_blocks": self.checkpoint_every,
            "checkpointed_blocks": checkpointed_blocks,
            "activation_checkpointing": checkpointed_blocks > 0,
            "checkpoint_implementation": "torch.utils.checkpoint, non-reentrant",
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
            "samples_per_second": samples / elapsed,
            "tokens_per_second": tokens / elapsed,
            "final_loss_mean": mean_loss,
            "steady_state_allocated_gib": steady_state_allocated / GIBIBYTE,
            "peak_allocated_gib": peak_allocated / GIBIBYTE,
            "peak_reserved_gib": peak_reserved / GIBIBYTE,
            "parameter_checksum_max_difference": checksum_difference,
            "parameters_synchronized": checksum_difference == 0.0,
            "gpu_name_rank_zero": torch.cuda.get_device_name(self.device),
            "hostname_rank_zero": socket.gethostname(),
            "pytorch_version": torch.__version__,
            "lightning_version": L.__version__,
            "cuda_version": torch.version.cuda,
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


def print_summary(result: dict[str, Any]) -> None:
    """Print a compact summary suitable for a Slurm log."""

    print("Lightning activation-checkpointing benchmark")
    print(f"  checkpoint interval:  {result['checkpoint_every_blocks']}")
    print(f"  checkpointed blocks:  {result['checkpointed_blocks']}")
    print(f"  preset:               {result['preset']}")
    print(f"  GPUs:                 {result['world_size']}")
    print(f"  precision:            {result['precision_resolved']}")
    print(f"  batch/device:         {result['batch_size_per_device']}")
    print(f"  time/step:            {result['milliseconds_per_step']:.3f} ms")
    print(f"  throughput:           {result['tokens_per_second']:.1f} tokens/s")
    print(f"  steady allocation:    {result['steady_state_allocated_gib']:.2f} GiB")
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
    benchmark = ActivationCheckpointBenchmark(
        config=PRESETS[arguments.preset],
        preset=arguments.preset,
        checkpoint_every=arguments.checkpoint_every,
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
    trainer.fit(benchmark)


if __name__ == "__main__":
    main()
