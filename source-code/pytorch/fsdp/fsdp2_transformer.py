#!/usr/bin/env python3
"""Compare native PyTorch DDP and FSDP2 on a synthetic transformer.

The model and token batches are created locally, so the benchmark downloads
nothing and measures model computation plus distributed communication rather
than storage or data-pipeline performance. The default ``small`` preset is
intended as a quick smoke test; ``medium`` makes model-state memory sharding
more visible on typical multi-GPU training nodes.

Run both strategies with the same number of processes and settings::

    torchrun --standalone --nproc-per-node=4 fsdp2_transformer.py \
        --strategy ddp --preset medium --output result-ddp.json
    torchrun --standalone --nproc-per-node=4 fsdp2_transformer.py \
        --strategy fsdp2 --preset medium --output result-fsdp2.json

FSDP2 is most useful for reducing model-state memory. It may be slower than
DDP when the unsharded model already fits comfortably on every GPU.
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

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP

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
        layers=12,
        hidden_size=768,
        attention_heads=12,
        feedforward_size=3072,
        sequence_length=256,
    ),
    "medium": ModelConfig(
        layers=24,
        hidden_size=1024,
        attention_heads=16,
        feedforward_size=4096,
        sequence_length=256,
    ),
    "large": ModelConfig(
        layers=32,
        hidden_size=1280,
        attention_heads=20,
        feedforward_size=5120,
        sequence_length=256,
    ),
}


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_arguments() -> argparse.Namespace:
    """Parse arguments and reject invalid output behavior before GPU setup."""

    parser = argparse.ArgumentParser(
        description="Compare DDP and FSDP2 with a synthetic transformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--strategy", choices=("ddp", "fsdp2"), required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument(
        "--fsdp-wrap",
        choices=("block", "root"),
        default="block",
        help="FSDP2 communication-group boundary; ignored for DDP",
    )
    batches = parser.add_mutually_exclusive_group()
    batches.add_argument(
        "--batch-size-per-device",
        type=positive_integer,
        default=1,
        help="samples per GPU; keeping this fixed is weak scaling",
    )
    batches.add_argument(
        "--global-batch-size",
        type=positive_integer,
        help="total samples across GPUs; must be divisible by WORLD_SIZE",
    )
    parser.add_argument(
        "--warmup-steps", type=positive_integer, default=5, help="untimed steps"
    )
    parser.add_argument(
        "--steps", type=positive_integer, default=15, help="timed optimizer steps"
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
            "example, 'torchrun --standalone --nproc-per-node=2 "
            "fsdp2_transformer.py --strategy fsdp2'"
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
    """Encoder-style transformer for a synthetic sequence classification task."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
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

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return sequence-classification logits for integer token IDs."""

        hidden_states = self.token_embedding(token_ids)
        hidden_states = hidden_states + self.position_embedding.unsqueeze(0)
        for block in self.blocks:
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


def configure_distributed_model(
    model: SyntheticTransformer,
    strategy: str,
    fsdp_wrap: str,
    use_bf16: bool,
    local_rank: int,
    device: torch.device,
) -> nn.Module:
    """Wrap the model with DDP or apply FSDP2 in place."""

    if strategy == "ddp":
        model.to(device)
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
        )

    mixed_precision_policy = (
        MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        if use_bf16
        else MixedPrecisionPolicy()
    )
    if fsdp_wrap == "block":
        for block in model.blocks:
            fully_shard(block, mp_policy=mixed_precision_policy)
    fully_shard(model, mp_policy=mixed_precision_policy)
    return model


def local_parameter_elements(model: nn.Module) -> int:
    """Count parameter elements physically held by this rank when sharded."""

    total = 0
    for parameter in model.parameters():
        if isinstance(parameter, DTensor):
            total += parameter.to_local().numel()
        else:
            total += parameter.numel()
    return total


def training_step(
    model: nn.Module,
    token_ids: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    use_ddp_autocast: bool,
) -> Tensor:
    """Run one forward, backward, communication, and optimizer cycle."""

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_ddp_autocast
    ):
        logits = model(token_ids)
        loss = nn.functional.cross_entropy(logits.float(), targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def parameter_checksums(
    model: nn.Module, strategy: str, device: torch.device
) -> tuple[float, float, float | None]:
    """Return global parameter checksums and DDP replica disagreement."""

    checksum = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for parameter in model.parameters():
            values = (
                parameter.to_local() if isinstance(parameter, DTensor) else parameter
            )
            values = values.detach().to(torch.float64)
            checksum[0] += values.sum()
            checksum[1] += values.square().sum()

    if strategy == "fsdp2":
        dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
        return checksum[0].item(), checksum[1].item(), None

    minimum = checksum.clone()
    maximum = checksum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    difference = (maximum - minimum).abs().max().item()
    return checksum[0].item(), checksum[1].item(), difference


def run(arguments: argparse.Namespace) -> dict[str, object]:
    """Run the selected strategy and return one benchmark result record."""

    local_rank, rank, world_size, device = initialize_distributed()
    per_device_batch, global_batch, batch_mode = resolve_batch_sizes(
        arguments, world_size
    )
    precision, use_bf16 = resolve_precision(arguments.precision, device)
    config = PRESETS[arguments.preset]

    torch.manual_seed(arguments.seed)
    model = SyntheticTransformer(config)
    global_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model = configure_distributed_model(
        model=model,
        strategy=arguments.strategy,
        fsdp_wrap=arguments.fsdp_wrap,
        use_bf16=use_bf16,
        local_rank=local_rank,
        device=device,
    )
    model.train()
    local_parameter_count = local_parameter_elements(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    token_ids, targets = make_batch(
        config, per_device_batch, rank, arguments.seed, device
    )

    use_ddp_autocast = arguments.strategy == "ddp" and use_bf16
    loss = torch.zeros((), device=device)
    for _ in range(arguments.warmup_steps):
        loss = training_step(model, token_ids, targets, optimizer, use_ddp_autocast)

    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for _ in range(arguments.steps):
        loss = training_step(model, token_ids, targets, optimizer, use_ddp_autocast)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

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
            "the final loss is not finite; retry with --precision fp32 or the "
            "small preset"
        )
    checksum_sum, checksum_squared_sum, checksum_difference = parameter_checksums(
        model, arguments.strategy, device
    )
    if not math.isfinite(checksum_sum) or not math.isfinite(checksum_squared_sum):
        raise RuntimeError("model parameters contain non-finite values")

    samples = arguments.steps * global_batch
    tokens = samples * config.sequence_length
    return {
        "model": "synthetic encoder-style transformer",
        "data": "synthetic GPU-resident integer token sequences",
        "strategy": arguments.strategy,
        "fsdp_wrap": (arguments.fsdp_wrap if arguments.strategy == "fsdp2" else None),
        "preset": arguments.preset,
        "model_config": asdict(config),
        "global_parameter_count": global_parameter_count,
        "local_parameter_elements": local_parameter_count,
        "local_parameter_fraction": (local_parameter_count / global_parameter_count),
        "model_parameters_fp32_gib": 4 * global_parameter_count / GIBIBYTE,
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
        "peak_allocated_gib": peak_allocated / GIBIBYTE,
        "peak_reserved_gib": peak_reserved / GIBIBYTE,
        "parameter_checksum_sum": checksum_sum,
        "parameter_checksum_squared_sum": checksum_squared_sum,
        "parameter_checksum_max_difference": checksum_difference,
        "parameters_synchronized": (
            checksum_difference == 0.0 if checksum_difference is not None else None
        ),
        "gpu_name_rank_zero": torch.cuda.get_device_name(device),
        "hostname_rank_zero": socket.gethostname(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def print_summary(result: dict[str, object]) -> None:
    """Print a compact summary suitable for a Slurm log."""

    print("DDP/FSDP2 synthetic transformer benchmark")
    print(f"  strategy:             {result['strategy']}")
    print(f"  FSDP wrap:            {result['fsdp_wrap']}")
    print(f"  preset:               {result['preset']}")
    print(f"  GPUs:                 {result['world_size']}")
    print(f"  parameters:           {result['global_parameter_count']:,}")
    print(f"  local param fraction: {result['local_parameter_fraction']:.3f}")
    print(f"  precision:            {result['precision_resolved']}")
    print(f"  time/step:            {result['milliseconds_per_step']:.3f} ms")
    print(f"  throughput:           {result['tokens_per_second']:.1f} tokens/s")
    print(f"  peak allocated/GPU:   {result['peak_allocated_gib']:.2f} GiB")
    print(f"  peak reserved/GPU:    {result['peak_reserved_gib']:.2f} GiB")
    if result["parameters_synchronized"] is not None:
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
