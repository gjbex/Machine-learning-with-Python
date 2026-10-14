#!/usr/bin/env python3
"""Collect phase-by-phase CUDA memory evidence for three mystery workloads.

The script runs on one GPU with plain ``python`` and optionally on several GPUs
with ``torchrun``.  It uses a locally constructed transformer and synthetic
GPU-resident inputs, so it downloads neither a model nor a dataset.

Start with one GPU and compare the anonymous cases empirically::

    python memory_diagnostics.py --case alpha --output alpha.json
    python memory_diagnostics.py --case beta --output beta.json
    python memory_diagnostics.py --case gamma --output gamma.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

MEBIBYTE = 2**20


@dataclass(frozen=True)
class ModelConfig:
    """Dimensions for one synthetic transformer preset."""

    layers: int
    hidden_size: int
    attention_heads: int
    feedforward_size: int
    sequence_length: int
    vocabulary_size: int = 4096
    classes: int = 16


@dataclass(frozen=True)
class CaseConfig:
    """Hidden implementation choices behind one mystery case."""

    optimizer: str
    retain_training_graphs: bool


PRESETS = {
    "small": ModelConfig(
        layers=4,
        hidden_size=384,
        attention_heads=6,
        feedforward_size=1536,
        sequence_length=256,
    ),
    "medium": ModelConfig(
        layers=8,
        hidden_size=512,
        attention_heads=8,
        feedforward_size=2048,
        sequence_length=512,
    ),
}

# The labels deliberately do not reveal the expected diagnosis. Participants
# should inspect this mapping only after using the measurements to form a
# hypothesis.
CASES = {
    "alpha": CaseConfig(optimizer="sgd", retain_training_graphs=False),
    "beta": CaseConfig(optimizer="adamw", retain_training_graphs=False),
    "gamma": CaseConfig(optimizer="sgd", retain_training_graphs=True),
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
    """Parse options and protect output files before allocating GPU memory."""

    parser = argparse.ArgumentParser(
        description="Diagnose CUDA memory pressure from phase measurements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument(
        "--batch-size-per-device",
        type=positive_integer,
        default=2,
        help="synthetic samples held on each GPU",
    )
    parser.add_argument(
        "--steps", type=positive_integer, default=6, help="measured training steps"
    )
    parser.add_argument(
        "--rank-zero-extra-samples",
        type=non_negative_integer,
        default=0,
        help="optional DDP extension that makes rank zero's local batch larger",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="BF16 requires support on every participating GPU",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output", type=Path, required=True, help="rank-zero JSON report path"
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="allow replacement of an existing JSON report",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="optional base path for a detailed allocator snapshot per rank",
    )
    parser.add_argument(
        "--snapshot-max-entries",
        type=positive_integer,
        default=100_000,
        help="maximum recorded allocator-history entries per rank",
    )
    parser.add_argument(
        "--overwrite-snapshot",
        action="store_true",
        help="allow replacement of rank-qualified snapshot files",
    )
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()

    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    validate_new_file(parser, arguments.output, arguments.overwrite_output)
    if arguments.snapshot is not None:
        validate_new_file(parser, arguments.snapshot, arguments.overwrite_snapshot)
    return arguments


def validate_new_file(
    parser: argparse.ArgumentParser, path: Path, overwrite: bool
) -> None:
    """Reject missing parents, directories, and accidental replacement."""

    if path.is_dir():
        parser.error(f"output path is a directory: {path}")
    if path.exists() and not overwrite:
        parser.error(f"output already exists: {path}; enable overwrite explicitly")
    if not path.parent.exists():
        parser.error(f"output directory does not exist: {path.parent}")


def initialize_runtime() -> tuple[int, int, int, torch.device]:
    """Select one CUDA device and initialize NCCL when launched by torchrun."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this exercise requires a GPU")

    distributed_variables = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    supplied = [name in os.environ for name in distributed_variables]
    if any(supplied) and not all(supplied):
        raise RuntimeError(
            "incomplete torchrun environment: LOCAL_RANK, RANK, and WORLD_SIZE "
            "must either all be set or all be absent"
        )

    if all(supplied):
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = rank = 0
        world_size = 1

    visible_devices = torch.cuda.device_count()
    if not 0 <= local_rank < visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank}, but this process sees {visible_devices} GPU(s)"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=timedelta(minutes=5)
        )
    return local_rank, rank, world_size, device


def resolve_precision(requested: str, device: torch.device) -> bool:
    """Return whether BF16 autocast can be used consistently across ranks."""

    if requested == "fp32":
        return False
    supported = torch.tensor(
        int(torch.cuda.is_bf16_supported()), dtype=torch.int32, device=device
    )
    if dist.is_initialized():
        dist.all_reduce(supported, op=dist.ReduceOp.MIN)
    if not bool(supported.item()):
        raise ValueError("BF16 was requested but at least one GPU lacks support")
    return True


class TransformerBlock(nn.Module):
    """A pre-normalized attention and feed-forward block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.attention_heads,
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
        """Apply attention and feed-forward residual paths."""

        normalized = self.attention_norm(hidden_states)
        attention_output, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden_states = hidden_states + attention_output
        return hidden_states + self.feedforward(self.feedforward_norm(hidden_states))


class SyntheticTransformer(nn.Module):
    """A compact encoder-style transformer for memory experiments."""

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
        """Return classification logits for token sequences."""

        hidden_states = self.token_embedding(token_ids)
        hidden_states = hidden_states + self.position_embedding.unsqueeze(0)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.output_norm(hidden_states[:, 0])
        return self.classifier(hidden_states)


def make_model(
    config: ModelConfig, local_rank: int, world_size: int, device: torch.device
) -> nn.Module:
    """Create the transformer and optionally wrap one replica in DDP."""

    model = SyntheticTransformer(config).to(device)
    if world_size == 1:
        return model
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
    )


def make_optimizer(
    model: nn.Module, case: CaseConfig
) -> torch.optim.Optimizer:
    """Create the optimizer selected by the anonymous workload."""

    if case.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=1e-3)
    if case.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=1e-4)
    raise AssertionError(f"unknown optimizer: {case.optimizer}")


def make_batch(
    config: ModelConfig,
    batch_size: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Create one reproducible rank-specific batch directly on the GPU."""

    generator = torch.Generator(device=device).manual_seed(seed + rank)
    token_ids = torch.randint(
        config.vocabulary_size,
        (batch_size, config.sequence_length),
        generator=generator,
        device=device,
    )
    targets = torch.randint(
        config.classes,
        (batch_size,),
        generator=generator,
        device=device,
    )
    return token_ids, targets


def sample_memory(device: torch.device, step: int, phase: str) -> dict[str, Any]:
    """Synchronize and record process-local and device-wide CUDA memory."""

    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "step": step,
        "phase": phase,
        "allocated_mib": torch.cuda.memory_allocated(device) / MEBIBYTE,
        "reserved_mib": torch.cuda.memory_reserved(device) / MEBIBYTE,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MEBIBYTE,
        "device_used_mib": (total_bytes - free_bytes) / MEBIBYTE,
    }


def training_step(
    model: nn.Module,
    token_ids: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    case: CaseConfig,
    use_bf16: bool,
    step: int,
    device: torch.device,
    retained_losses: list[Tensor],
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Run one cycle and measure forward, backward, update, and cleanup phases."""

    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(token_ids)
        loss = nn.functional.cross_entropy(logits.float(), targets)
    samples = [sample_memory(device, step, "after_forward")]

    # One anonymous case deliberately preserves old training graphs. The
    # measurements, rather than this line, should be the participant's first
    # evidence that live allocations accumulate between steps.
    loss.backward(retain_graph=case.retain_training_graphs)
    samples.append(sample_memory(device, step, "after_backward"))
    optimizer.step()
    samples.append(sample_memory(device, step, "after_optimizer_step"))

    if case.retain_training_graphs:
        retained_losses.append(loss)
    optimizer.zero_grad(set_to_none=True)
    samples.append(sample_memory(device, step, "after_clear_gradients"))
    return loss.detach(), samples


def snapshot_path(base: Path, rank: int) -> Path:
    """Make allocator snapshot names collision-free across ranks."""

    suffix = base.suffix or ".pickle"
    stem = base.stem if base.suffix else base.name
    return base.with_name(f"{stem}-rank{rank}{suffix}")


def enable_memory_history(max_entries: int) -> None:
    """Enable PyTorch's optional allocator-history recorder."""

    memory_module = torch.cuda.memory
    recorder = getattr(memory_module, "_record_memory_history", None)
    if recorder is None:
        raise RuntimeError(
            "this PyTorch build does not provide CUDA allocator history recording"
        )
    recorder(max_entries=max_entries)


def dump_memory_snapshot(path: Path) -> None:
    """Write a snapshot consumable by the PyTorch memory visualizer."""

    dumper = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if dumper is None:
        raise RuntimeError("this PyTorch build cannot dump CUDA memory snapshots")
    dumper(str(path))


def derive_indicators(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Derive diagnosis aids without assigning a semantic label to the case."""

    initial = next(sample for sample in samples if sample["phase"] == "after_input")
    cleared = [
        sample for sample in samples if sample["phase"] == "after_clear_gradients"
    ]
    forwards = [sample for sample in samples if sample["phase"] == "after_forward"]
    first_persistent_change = cleared[0]["allocated_mib"] - initial["allocated_mib"]
    between_step_growth = cleared[-1]["allocated_mib"] - cleared[0]["allocated_mib"]

    baselines = [initial, *cleared[:-1]]
    forward_increases = [
        forward["allocated_mib"] - baseline["allocated_mib"]
        for forward, baseline in zip(forwards, baselines, strict=True)
    ]
    return {
        "first_step_persistent_change_mib": first_persistent_change,
        "between_step_growth_mib": between_step_growth,
        "largest_forward_increase_mib": max(forward_increases),
        "maximum_peak_allocated_mib": max(
            sample["peak_allocated_mib"] for sample in samples
        ),
    }


def gather_rank_reports(
    local_report: dict[str, Any], rank: int, world_size: int
) -> list[dict[str, Any]] | None:
    """Collect compact per-rank evidence on rank zero."""

    if world_size == 1:
        return [local_report]
    gathered: list[dict[str, Any] | None] | None
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local_report, gathered, dst=0)
    if rank != 0:
        return None
    assert gathered is not None and all(report is not None for report in gathered)
    return [report for report in gathered if report is not None]


def run(arguments: argparse.Namespace) -> dict[str, Any] | None:
    """Construct the workload, collect phase evidence, and gather all ranks."""

    local_rank, rank, world_size, device = initialize_runtime()
    use_bf16 = resolve_precision(arguments.precision, device)
    config = PRESETS[arguments.preset]
    case = CASES[arguments.case]
    rank_snapshot = (
        snapshot_path(arguments.snapshot, rank)
        if arguments.snapshot is not None
        else None
    )
    if rank_snapshot is not None and rank_snapshot.exists():
        if not arguments.overwrite_snapshot:
            raise FileExistsError(
                f"snapshot already exists: {rank_snapshot}; enable overwrite explicitly"
            )

    if rank_snapshot is not None:
        enable_memory_history(arguments.snapshot_max_entries)

    torch.manual_seed(arguments.seed)
    model = make_model(config, local_rank, world_size, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)
    samples = [sample_memory(device, -1, "after_model")]

    optimizer = make_optimizer(model, case)
    samples.append(sample_memory(device, -1, "after_optimizer_creation"))
    token_ids, targets = make_batch(
        config,
        arguments.batch_size_per_device
        + (arguments.rank_zero_extra_samples if rank == 0 else 0),
        rank,
        arguments.seed,
        device,
    )
    samples.append(sample_memory(device, -1, "after_input"))

    retained_losses: list[Tensor] = []
    loss = torch.zeros((), device=device)
    for step in range(arguments.steps):
        loss, step_samples = training_step(
            model,
            token_ids,
            targets,
            optimizer,
            case,
            use_bf16,
            step,
            device,
            retained_losses,
        )
        samples.extend(step_samples)

    if not math.isfinite(loss.item()):
        raise RuntimeError("the final loss is not finite; retry with FP32")
    if rank_snapshot is not None:
        dump_memory_snapshot(rank_snapshot)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    local_report = {
        "rank": rank,
        "local_rank": local_rank,
        "local_batch_size": token_ids.shape[0],
        "gpu_name": torch.cuda.get_device_name(device),
        "device_total_mib": total_bytes / MEBIBYTE,
        "device_free_at_end_mib": free_bytes / MEBIBYTE,
        "snapshot": str(rank_snapshot) if rank_snapshot is not None else None,
        "samples": samples,
        "indicators": derive_indicators(samples),
    }
    rank_reports = gather_rank_reports(local_report, rank, world_size)
    if rank != 0:
        return None
    assert rank_reports is not None
    return {
        "exercise": "empirical CUDA memory diagnosis",
        "case": arguments.case,
        "preset": arguments.preset,
        "model_config": asdict(config),
        "parameter_count": parameter_count,
        "parameters_fp32_mib": 4 * parameter_count / MEBIBYTE,
        "batch_size_per_device": arguments.batch_size_per_device,
        "rank_zero_extra_samples": arguments.rank_zero_extra_samples,
        "steps": arguments.steps,
        "precision": arguments.precision,
        "world_size": world_size,
        "seed": arguments.seed,
        "final_loss_rank_zero": loss.item(),
        "rank_reports": rank_reports,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "hostname_rank_zero": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "measurement_notes": {
            "allocated": "live tensor memory tracked by the PyTorch CUDA allocator",
            "reserved": "memory held by the PyTorch caching allocator",
            "device_used": "device-wide use, including other processes and libraries",
            "peak": "maximum allocated since the per-step peak reset",
        },
    }


def print_summary(result: dict[str, Any]) -> None:
    """Print evidence from every rank in a compact Slurm-log format."""

    print("PyTorch CUDA memory-diagnostics exercise")
    print(f"  mystery case:          {result['case']}")
    print(f"  preset:                {result['preset']}")
    print(f"  GPUs:                  {result['world_size']}")
    print(f"  batch/device:          {result['batch_size_per_device']}")
    print(f"  precision:             {result['precision']}")
    print(f"  FP32 parameter memory: {result['parameters_fp32_mib']:.1f} MiB")
    for rank_report in result["rank_reports"]:
        indicators = rank_report["indicators"]
        print(f"  rank {rank_report['rank']} indicators:")
        print(
            "    first-step persistent change: "
            f"{indicators['first_step_persistent_change_mib']:.1f} MiB"
        )
        print(
            "    between-step growth:          "
            f"{indicators['between_step_growth_mib']:.1f} MiB"
        )
        print(
            "    largest forward increase:     "
            f"{indicators['largest_forward_increase_mib']:.1f} MiB"
        )
        print(
            "    maximum peak allocated:       "
            f"{indicators['maximum_peak_allocated_mib']:.1f} MiB"
        )


def main() -> int:
    """Run the selected case and write the rank-zero JSON report."""

    arguments = parse_arguments()
    try:
        result = run(arguments)
        if result is not None:
            print_summary(result)
            text = json.dumps(result, indent=2, sort_keys=True)
            arguments.output.write_text(f"{text}\n", encoding="utf-8")
            print(f"  JSON result:           {arguments.output}")
        return 0
    except torch.cuda.OutOfMemoryError as error:
        rank = os.environ.get("RANK", "0")
        print(
            f"rank {rank}: CUDA out of memory. This is capacity evidence, but the "
            "exercise does not require an OOM. Retry with --preset small or a "
            f"smaller batch. Original error: {error}",
            file=sys.stderr,
        )
        return 3
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        rank = os.environ.get("RANK", "0")
        print(f"rank {rank}: error: {error}", file=sys.stderr)
        return 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
