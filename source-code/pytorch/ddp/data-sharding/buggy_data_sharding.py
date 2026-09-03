#!/usr/bin/env python3
"""Diagnose an intentionally incorrect DDP input pipeline.

Run this program with at least two processes.  It trains a tiny classifier and
records the sample identifiers processed by every rank.  The input pipeline is
plausible for single-process training but intentionally incorrect for DDP.

Example::

    torchrun --standalone --nproc-per-node=2 buggy_data_sharding.py \
        --device cpu --output buggy-report.json

The program exits successfully when it exposes the intended data-sharding
failure.  That makes it convenient to use in a scheduled hands-on exercise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DistributedRuntime:
    """Information about this process in the distributed job."""

    local_rank: int
    rank: int
    world_size: int
    device: torch.device
    backend: str


class IndexedClassificationDataset(Dataset[tuple[Tensor, Tensor, int]]):
    """Small deterministic dataset that exposes each sample's identifier."""

    def __init__(
        self,
        size: int,
        feature_count: int,
        class_count: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(size, feature_count, generator=generator)
        projection = torch.randn(feature_count, class_count, generator=generator)
        self.targets = (self.features @ projection).argmax(dim=1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        return self.features[index], self.targets[index], index


def positive_integer(value: str) -> int:
    """Parse a positive command-line integer."""

    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_arguments() -> argparse.Namespace:
    """Parse and validate options that do not depend on the worker count."""

    parser = argparse.ArgumentParser(
        description="Expose a data-sharding bug in a native PyTorch DDP job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-size", type=positive_integer, default=32)
    parser.add_argument("--batch-size", type=positive_integer, default=4)
    parser.add_argument("--epochs", type=positive_integer, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when available and CPU otherwise",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path written by rank zero",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="allow replacement of an existing JSON report",
    )
    parser.add_argument(
        "--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS
    )
    arguments = parser.parse_args()

    if arguments.dataset_size > 256:
        parser.error("--dataset-size must not exceed 256 in this lab exercise")
    if arguments.epochs < 2:
        parser.error("--epochs must be at least 2 to inspect epoch reshuffling")
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if arguments.output is None and arguments.overwrite_output:
        parser.error("--overwrite-output requires --output")
    if arguments.output is not None:
        if arguments.output.is_dir():
            parser.error(f"output path is a directory: {arguments.output}")
        if arguments.output.exists() and not arguments.overwrite_output:
            parser.error(
                f"output already exists: {arguments.output}; use "
                "--overwrite-output to replace it"
            )
        if not arguments.output.parent.exists():
            parser.error(f"output directory does not exist: {arguments.output.parent}")
    return arguments


def initialize_distributed(requested_device: str) -> DistributedRuntime:
    """Initialize a torchrun process group and select this rank's device."""

    required = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"missing torchrun variables {', '.join(missing)}; launch with "
            "'torchrun --standalone --nproc-per-node=2 "
            "buggy_data_sharding.py'"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError("this exercise requires at least two processes")

    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and torch.cuda.is_available()
    )
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is unavailable")

    if use_cuda:
        visible_devices = torch.cuda.device_count()
        if not 0 <= local_rank < visible_devices:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but this process sees "
                f"{visible_devices} CUDA device(s)"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(
        backend=backend, init_method="env://", timeout=timedelta(minutes=5)
    )
    return DistributedRuntime(local_rank, rank, world_size, device, backend)


def validate_distributed_arguments(
    arguments: argparse.Namespace, runtime: DistributedRuntime
) -> None:
    """Reject configurations that make the sharding evidence ambiguous."""

    if arguments.dataset_size % runtime.world_size != 0:
        raise ValueError(
            f"--dataset-size={arguments.dataset_size} must be divisible by "
            f"WORLD_SIZE={runtime.world_size}; otherwise DistributedSampler "
            "pads its index list and the reference run can contain duplicates"
        )
    intended_samples_per_rank = arguments.dataset_size // runtime.world_size
    if arguments.batch_size > intended_samples_per_rank:
        raise ValueError(
            f"--batch-size={arguments.batch_size} exceeds the intended "
            f"per-rank shard size {intended_samples_per_rank}"
        )


def make_model(feature_count: int, class_count: int) -> nn.Module:
    """Create the deliberately small classifier used by the exercise."""

    return nn.Sequential(
        nn.Linear(feature_count, 32),
        nn.ReLU(),
        nn.Linear(32, class_count),
    )


def make_dataloader(
    dataset: IndexedClassificationDataset,
    batch_size: int,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    """Create the intentionally incorrect input pipeline for each rank."""

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def train_one_epoch(
    model: DDP,
    dataloader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, list[int]]:
    """Train for one epoch and return global loss plus local sample IDs."""

    model.train()
    local_loss_sum = 0.0
    local_example_count = 0
    local_sample_ids: list[int] = []
    non_blocking = device.type == "cuda"

    for features, targets, sample_ids in dataloader:
        features = features.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features), targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.shape[0]
        local_loss_sum += loss.detach().item() * batch_size
        local_example_count += batch_size
        local_sample_ids.extend(int(index) for index in sample_ids.tolist())

    totals = torch.tensor(
        [local_loss_sum, local_example_count],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return (totals[0] / totals[1]).item(), local_sample_ids


def gather_sample_ids(
    local_sample_ids: list[int], runtime: DistributedRuntime
) -> list[list[int]] | None:
    """Gather each rank's ordered sample identifiers on rank zero."""

    gathered: list[Any] | None = (
        [None] * runtime.world_size if runtime.rank == 0 else None
    )
    dist.gather_object(local_sample_ids, gathered, dst=0)
    if gathered is None:
        return None
    if not all(isinstance(item, list) for item in gathered):
        raise RuntimeError("rank zero did not receive every sample-ID list")
    return [[int(index) for index in item] for item in gathered]


def analyze_assignments(
    sample_ids_by_rank: list[list[int]], dataset_size: int, epoch: int
) -> dict[str, Any]:
    """Measure coverage, duplication, and overlap in one epoch."""

    flattened = [
        sample_id
        for rank_sample_ids in sample_ids_by_rank
        for sample_id in rank_sample_ids
    ]
    counts = Counter(flattened)
    expected = set(range(dataset_size))
    observed = set(flattened)
    overlaps = []
    for left_rank, left_ids in enumerate(sample_ids_by_rank):
        for right_rank in range(left_rank + 1, len(sample_ids_by_rank)):
            shared = sorted(set(left_ids) & set(sample_ids_by_rank[right_rank]))
            overlaps.append(
                {
                    "ranks": [left_rank, right_rank],
                    "shared_count": len(shared),
                    "shared_sample_ids": shared,
                }
            )

    return {
        "epoch": epoch,
        "rank_sample_ids": sample_ids_by_rank,
        "processed_assignments": len(flattened),
        "unique_samples": len(observed),
        "duplicate_assignments": sum(count - 1 for count in counts.values()),
        "duplicated_sample_ids": sorted(
            sample_id for sample_id, count in counts.items() if count > 1
        ),
        "missing_sample_ids": sorted(expected - observed),
        "pairwise_rank_overlap": overlaps,
        "rank_orders_identical": all(
            rank_ids == sample_ids_by_rank[0] for rank_ids in sample_ids_by_rank[1:]
        ),
        "exactly_once_global": len(flattened) == dataset_size and observed == expected,
    }


def print_epoch_report(report: dict[str, Any]) -> None:
    """Print concise, human-readable evidence for one epoch."""

    print(f"\nepoch {report['epoch']}")
    for rank, sample_ids in enumerate(report["rank_sample_ids"]):
        print(f"  rank {rank}: {sample_ids}")
    print(
        "  assignments: "
        f"{report['processed_assignments']}; unique samples: "
        f"{report['unique_samples']}; duplicate assignments: "
        f"{report['duplicate_assignments']}"
    )
    print(f"  missing sample IDs: {report['missing_sample_ids']}")
    print(f"  exactly once globally: {report['exactly_once_global']}")


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a machine-readable report from rank zero."""

    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run(arguments: argparse.Namespace) -> None:
    """Set up DDP, build the input pipeline, and run the diagnosis."""

    runtime = initialize_distributed(arguments.device)
    validate_distributed_arguments(arguments, runtime)

    feature_count = 16
    class_count = 4
    torch.manual_seed(arguments.seed)
    dataset = IndexedClassificationDataset(
        arguments.dataset_size, feature_count, class_count, arguments.seed
    )
    dataloader = make_dataloader(dataset, arguments.batch_size, arguments.seed)

    model = make_model(feature_count, class_count).to(runtime.device)
    if runtime.device.type == "cuda":
        distributed_model = DDP(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
        )
    else:
        distributed_model = DDP(model)
    optimizer = torch.optim.SGD(distributed_model.parameters(), lr=0.05)

    epoch_reports: list[dict[str, Any]] = []
    losses: list[float] = []
    for epoch in range(arguments.epochs):
        mean_loss, local_sample_ids = train_one_epoch(
            distributed_model, dataloader, optimizer, runtime.device
        )
        sample_ids_by_rank = gather_sample_ids(local_sample_ids, runtime)
        if runtime.rank == 0:
            if sample_ids_by_rank is None:
                raise RuntimeError("rank zero did not receive sample identifiers")
            epoch_report = analyze_assignments(
                sample_ids_by_rank, arguments.dataset_size, epoch
            )
            epoch_reports.append(epoch_report)
            losses.append(mean_loss)
            print_epoch_report(epoch_report)

    if runtime.rank != 0:
        return

    rank_order_changed = [
        any(
            epoch_reports[epoch]["rank_sample_ids"][rank]
            != epoch_reports[0]["rank_sample_ids"][rank]
            for epoch in range(1, len(epoch_reports))
        )
        for rank in range(runtime.world_size)
    ]
    all_epochs_exactly_once = all(
        report["exactly_once_global"] for report in epoch_reports
    )
    report = {
        "implementation": "buggy_plain_dataloader",
        "world_size": runtime.world_size,
        "device": runtime.device.type,
        "backend": runtime.backend,
        "dataset_size": arguments.dataset_size,
        "batch_size_per_rank": arguments.batch_size,
        "epochs": arguments.epochs,
        "seed": arguments.seed,
        "mean_loss_by_epoch": losses,
        "rank_order_changed_between_epochs": rank_order_changed,
        "all_ranks_changed_order_between_epochs": all(rank_order_changed),
        "all_epochs_exactly_once": all_epochs_exactly_once,
        "epoch_reports": epoch_reports,
        "pytorch_version": torch.__version__,
    }
    if arguments.output is not None:
        write_report(arguments.output, report)
        print(f"\nwrote {arguments.output}")

    if all_epochs_exactly_once:
        print("\nSHARDING CHECK: unexpectedly passed")
    else:
        print("\nSHARDING CHECK: failed as intended; diagnose the evidence above")


def main() -> None:
    """Run the exercise and always release the process group."""

    try:
        run(parse_arguments())
    except (OSError, RuntimeError, ValueError) as error:
        rank = os.environ.get("RANK", "unknown")
        print(f"rank {rank}: error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
