#!/usr/bin/env python3
"""Measure the memory/time trade-off of activation checkpointing with DDP.

The model and batches are synthetic and are created locally. Compare otherwise
identical runs with ``--checkpoint-every 0`` and ``--checkpoint-every 1``::

    torchrun --standalone --nproc-per-node=1 activation_checkpointing.py \
        --checkpoint-every 0 --output result-no-checkpoint.json
    torchrun --standalone --nproc-per-node=1 activation_checkpointing.py \
        --checkpoint-every 1 --output result-checkpoint.json

Activation checkpointing discards selected intermediate activations during the
forward pass and recomputes them during backward. It does not shard parameters,
gradients, or optimizer state.
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
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

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
    """Parse arguments and reject invalid output behavior before GPU setup."""

    parser = argparse.ArgumentParser(
        description="Benchmark activation checkpointing with PyTorch DDP.",
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
        help="total samples across GPUs; must be divisible by WORLD_SIZE",
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
        help="auto uses BF16 when every GPU supports it, otherwise FP32",
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
    """Initialize NCCL and bind this process to its torchrun GPU."""

    required = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"missing torchrun variables {', '.join(missing)}; launch with, for "
            "example, 'torchrun --standalone --nproc-per-node=1 "
            "activation_checkpointing.py'"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires a GPU")

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
    """Resolve per-device and global batch sizes after distributed setup."""

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


def resolve_precision(requested: str, device: torch.device) -> tuple[str, bool]:
    """Choose FP32 or BF16 consistently across all participating GPUs."""

    bf16_supported = torch.tensor(
        int(torch.cuda.is_bf16_supported()), dtype=torch.int32, device=device
    )
    dist.all_reduce(bf16_supported, op=dist.ReduceOp.MIN)
    all_support_bf16 = bool(bf16_supported.item())

    if requested == "fp32":
        return "fp32", False
    if requested == "bf16":
        if not all_support_bf16:
            raise ValueError(
                "BF16 was requested, but at least one GPU does not support it"
            )
        return "bf16", True
    return ("bf16", True) if all_support_bf16 else ("fp32", False)


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


def make_batch(
    config: ModelConfig,
    batch_size: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Create one rank-specific, GPU-resident synthetic batch."""

    generator = torch.Generator(device=device).manual_seed(seed + rank)
    token_ids = torch.randint(
        0,
        config.vocabulary_size,
        (batch_size, config.sequence_length),
        generator=generator,
        device=device,
    )
    targets = torch.randint(
        0,
        config.classes,
        (batch_size,),
        generator=generator,
        device=device,
    )
    return token_ids, targets


def training_step(
    model: DDP,
    token_ids: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    use_bf16: bool,
) -> Tensor:
    """Run forward, backward, DDP synchronization, and the optimizer update."""

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(token_ids)
        loss = nn.functional.cross_entropy(logits.float(), targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def parameter_checksum_difference(model: nn.Module, device: torch.device) -> float:
    """Return the largest checksum difference between DDP replicas."""

    checksum = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for parameter in model.parameters():
            values = parameter.detach().to(torch.float64)
            checksum[0] += values.sum()
            checksum[1] += values.square().sum()
    minimum = checksum.clone()
    maximum = checksum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return (maximum - minimum).abs().max().item()


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run one activation-checkpoint configuration and return its metrics."""

    local_rank, rank, world_size, device = initialize_distributed()
    per_device_batch, global_batch, batch_mode = resolve_batch_sizes(
        arguments, world_size
    )
    precision, use_bf16 = resolve_precision(arguments.precision, device)
    config = PRESETS[arguments.preset]

    torch.manual_seed(arguments.seed)
    model = SyntheticTransformer(config, arguments.checkpoint_every).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpointed_blocks = sum(
        model.should_checkpoint(index) for index in range(config.layers)
    )
    distributed_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.SGD(distributed_model.parameters(), lr=1e-3)
    token_ids, targets = make_batch(
        config, per_device_batch, rank, arguments.seed, device
    )

    loss = torch.zeros((), device=device)
    for _ in range(arguments.warmup_steps):
        loss = training_step(distributed_model, token_ids, targets, optimizer, use_bf16)

    optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    torch.cuda.synchronize(device)
    steady_state_allocated = float(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for _ in range(arguments.steps):
        loss = training_step(distributed_model, token_ids, targets, optimizer, use_bf16)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    maxima = torch.tensor(
        [
            elapsed,
            steady_state_allocated,
            float(torch.cuda.max_memory_allocated(device)),
            float(torch.cuda.max_memory_reserved(device)),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    elapsed, steady_state_allocated, peak_allocated, peak_reserved = maxima.tolist()

    loss = loss.to(torch.float64)
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    mean_loss = loss.item() / world_size
    if not math.isfinite(mean_loss):
        raise RuntimeError(
            "the final loss is not finite; retry with --precision fp32 or the "
            "small preset"
        )
    checksum_difference = parameter_checksum_difference(distributed_model, device)

    samples = arguments.steps * global_batch
    tokens = samples * config.sequence_length
    return {
        "model": "synthetic encoder-style transformer",
        "data": "synthetic GPU-resident integer token sequences",
        "preset": arguments.preset,
        "model_config": asdict(config),
        "parameter_count": parameter_count,
        "model_parameters_fp32_gib": 4 * parameter_count / GIBIBYTE,
        "checkpoint_every_blocks": arguments.checkpoint_every,
        "checkpointed_blocks": checkpointed_blocks,
        "activation_checkpointing": checkpointed_blocks > 0,
        "checkpoint_implementation": "torch.utils.checkpoint, non-reentrant",
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
        "samples_per_second": samples / elapsed,
        "tokens_per_second": tokens / elapsed,
        "final_loss_mean": mean_loss,
        "steady_state_allocated_gib": steady_state_allocated / GIBIBYTE,
        "peak_allocated_gib": peak_allocated / GIBIBYTE,
        "peak_reserved_gib": peak_reserved / GIBIBYTE,
        "parameter_checksum_max_difference": checksum_difference,
        "parameters_synchronized": checksum_difference == 0.0,
        "gpu_name_rank_zero": torch.cuda.get_device_name(device),
        "hostname_rank_zero": socket.gethostname(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def print_summary(result: dict[str, Any]) -> None:
    """Print a compact summary suitable for a Slurm log."""

    print("PyTorch activation-checkpointing benchmark")
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
    except (OSError, RuntimeError, ValueError) as error:
        rank = os.environ.get("RANK", "unknown")
        print(f"rank {rank}: error: {error}", file=sys.stderr)
        return 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
