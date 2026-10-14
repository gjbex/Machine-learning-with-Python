#!/usr/bin/env python3
"""Compare Lightning DDP and FSDP2 on a synthetic transformer.

This is the Lightning counterpart of
``source-code/pytorch/fsdp/fsdp2_transformer.py``. It uses the same model,
synthetic token batches, optimizer, presets, warm-up period, measurements, and
JSON result fields. Lightning owns process setup, device placement, backward
propagation, and optimizer execution.

Inside a single-node Slurm allocation with four tasks and four GPUs::

    srun python fsdp2_transformer.py --strategy fsdp2 --devices 4 \
        --preset medium --output result-fsdp2.json

Outside Slurm, Lightning can launch the worker processes itself::

    python fsdp2_transformer.py --strategy fsdp2 --devices 4 \
        --preset medium --output result-fsdp2.json

No model, tokenizer, weights, or dataset are downloaded. FSDP2 is useful for
reducing model-state memory, but can be slower than DDP when the complete model
already fits comfortably on every GPU.
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

import lightning as L
import torch
from lightning.pytorch.strategies import DDPStrategy, ModelParallelStrategy
from torch import Tensor, nn
from torch.distributed import ReduceOp
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
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
    """Parse arguments and reject invalid runs before Trainer startup."""

    parser = argparse.ArgumentParser(
        description="Compare Lightning DDP and FSDP2 with a synthetic transformer.",
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
        help="total samples across GPUs; must be divisible by devices x nodes",
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
        help="auto uses BF16 when supported by the selected GPU, otherwise FP32",
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


def resolve_precision(requested: str) -> str:
    """Resolve the shared CLI precision before Lightning launches workers."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires GPUs")
    if requested == "fp32":
        return "fp32"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "BF16 was requested, but the selected GPU does not support it"
            )
        return "bf16"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp32"


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


class TransformerBenchmark(L.LightningModule):
    """Synthetic transformer workload with timing and reporting hooks."""

    def __init__(
        self,
        *,
        config: ModelConfig,
        preset: str,
        strategy_name: str,
        fsdp_wrap: str,
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
        self.config = config
        self.preset = preset
        self.strategy_name = strategy_name
        self.fsdp_wrap = fsdp_wrap
        self.batch_size_per_device_requested = batch_size_per_device
        self.global_batch_size_requested = global_batch_size
        self.warmup_steps = warmup_steps
        self.measured_steps = measured_steps
        self.precision_requested = precision_requested
        self.precision_resolved = precision_resolved
        self.seed = seed
        self.output = output

        # Large layers belong in configure_model(), which Lightning invokes in
        # a strategy-aware context. Keeping this hook idempotent is required.
        self.model: SyntheticTransformer | None = None
        self._global_parameter_count = 0
        self._local_parameter_count = 0
        self._token_ids: Tensor | None = None
        self._targets: Tensor | None = None
        self._last_loss: Tensor | None = None
        self._start_time: float | None = None
        self._maxima: Tensor | None = None
        self._batch_mode = ""
        self._batch_size_per_device = 0
        self._global_batch_size = 0

    def configure_model(self) -> None:
        """Create the model and apply FSDP2 through Lightning's device mesh."""

        if self.model is not None:
            return
        torch.manual_seed(self.seed)
        self.model = SyntheticTransformer(self.config)
        self._global_parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        if self.strategy_name != "fsdp2":
            return
        if self.device_mesh is None:
            raise RuntimeError("Lightning did not create the FSDP2 device mesh")

        data_parallel_mesh = self.device_mesh["data_parallel"]
        mixed_precision_policy = (
            MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
            if self.precision_resolved == "bf16"
            else MixedPrecisionPolicy()
        )
        if self.fsdp_wrap == "block":
            for block in self.model.blocks:
                fully_shard(
                    block,
                    mesh=data_parallel_mesh,
                    mp_policy=mixed_precision_policy,
                )
        fully_shard(
            self.model,
            mesh=data_parallel_mesh,
            mp_policy=mixed_precision_policy,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """Apply the configured model to integer token IDs."""

        if self.model is None:
            raise RuntimeError("model was not created by configure_model()")
        return self.model(token_ids)

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
        """Measure local shards and create one rank-specific synthetic batch."""

        if self.model is None:
            raise RuntimeError("model was not created by configure_model()")
        self._local_parameter_count = local_parameter_elements(self.model)
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
        """Create AdamW after configure_model() has applied any sharding."""

        if self.model is None:
            raise RuntimeError("model was not created before optimizer setup")
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3)

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

    def _parameter_checksums(self) -> tuple[float, float, float | None]:
        """Return global checksums and DDP replica disagreement when applicable."""

        if self.model is None:
            raise RuntimeError("model was not created by configure_model()")
        checksum = torch.zeros(2, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for parameter in self.model.parameters():
                values = (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                )
                values = values.detach().to(torch.float64)
                checksum[0] += values.sum()
                checksum[1] += values.square().sum()

        if self.strategy_name == "fsdp2":
            checksum = self.trainer.strategy.reduce(checksum, reduce_op=ReduceOp.SUM)
            return checksum[0].item(), checksum[1].item(), None

        minimum = self.trainer.strategy.reduce(checksum.clone(), reduce_op=ReduceOp.MIN)
        maximum = self.trainer.strategy.reduce(checksum, reduce_op=ReduceOp.MAX)
        difference = (maximum - minimum).abs().max().item()
        return checksum[0].item(), checksum[1].item(), difference

    def on_train_end(self) -> None:
        """Reduce correctness metrics and write one result on global rank zero."""

        if self._last_loss is None or self._maxima is None:
            raise RuntimeError("the benchmark did not complete all requested steps")

        mean_loss = self.trainer.strategy.reduce(
            self._last_loss.to(torch.float64), reduce_op="mean"
        ).item()
        if not math.isfinite(mean_loss):
            raise RuntimeError(
                "the final loss is not finite; retry with --precision fp32 or the "
                "small preset"
            )
        checksum_sum, checksum_squared_sum, checksum_difference = (
            self._parameter_checksums()
        )
        if not math.isfinite(checksum_sum) or not math.isfinite(checksum_squared_sum):
            raise RuntimeError("model parameters contain non-finite values")

        elapsed, peak_allocated, peak_reserved = self._maxima.tolist()
        samples = self.measured_steps * self._global_batch_size
        tokens = samples * self.config.sequence_length
        result = {
            "model": "synthetic encoder-style transformer",
            "data": "synthetic GPU-resident integer token sequences",
            "strategy": self.strategy_name,
            "fsdp_wrap": (self.fsdp_wrap if self.strategy_name == "fsdp2" else None),
            "preset": self.preset,
            "model_config": asdict(self.config),
            "global_parameter_count": self._global_parameter_count,
            "local_parameter_elements": self._local_parameter_count,
            "local_parameter_fraction": (
                self._local_parameter_count / self._global_parameter_count
            ),
            "model_parameters_fp32_gib": (4 * self._global_parameter_count / GIBIBYTE),
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
            "peak_allocated_gib": peak_allocated / GIBIBYTE,
            "peak_reserved_gib": peak_reserved / GIBIBYTE,
            "parameter_checksum_sum": checksum_sum,
            "parameter_checksum_squared_sum": checksum_squared_sum,
            "parameter_checksum_max_difference": checksum_difference,
            "parameters_synchronized": (
                checksum_difference == 0.0 if checksum_difference is not None else None
            ),
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


def local_parameter_elements(model: nn.Module) -> int:
    """Count parameter elements physically held by this rank when sharded."""

    total = 0
    for parameter in model.parameters():
        if isinstance(parameter, DTensor):
            total += parameter.to_local().numel()
        else:
            total += parameter.numel()
    return total


def print_summary(result: dict[str, object]) -> None:
    """Print a compact summary suitable for a Slurm log."""

    print("Lightning DDP/FSDP2 synthetic transformer benchmark")
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


def make_strategy(
    strategy_name: str, devices: int, num_nodes: int
) -> DDPStrategy | ModelParallelStrategy:
    """Create the Lightning strategy for replicated or FSDP2 training."""

    timeout = timedelta(minutes=5)
    if strategy_name == "ddp":
        return DDPStrategy(timeout=timeout, gradient_as_bucket_view=True)
    return ModelParallelStrategy(
        data_parallel_size=devices * num_nodes,
        tensor_parallel_size=1,
        timeout=timeout,
    )


def main() -> None:
    """Create the module and run the selected barebones Lightning Trainer."""

    arguments = parse_arguments()
    resolved_precision = resolve_precision(arguments.precision)
    trainer_precision = (
        "bf16-mixed"
        if arguments.strategy == "ddp" and resolved_precision == "bf16"
        else "32-true"
    )
    L.seed_everything(arguments.seed, workers=True)
    model = TransformerBenchmark(
        config=PRESETS[arguments.preset],
        preset=arguments.preset,
        strategy_name=arguments.strategy,
        fsdp_wrap=arguments.fsdp_wrap,
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
        strategy=make_strategy(
            arguments.strategy, arguments.devices, arguments.num_nodes
        ),
        precision=trainer_precision,
        max_steps=arguments.warmup_steps + arguments.steps,
        max_epochs=1,
        use_distributed_sampler=False,
        barebones=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
