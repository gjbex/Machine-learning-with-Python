# Exercise: diagnose DDP data sharding

This 30–40 minute exercise demonstrates a correctness problem that can be easy
to miss: native PyTorch `DistributedDataParallel` (DDP) synchronizes gradients,
but it does **not** automatically divide a dataset between ranks.

The exercise uses a tiny deterministic dataset whose samples have visible
integer identifiers. It therefore completes quickly, needs no downloads, and
lets you inspect data ownership directly instead of inferring it from training
speed or model accuracy.

## Learning objective and success criterion

After the exercise, you should be able to diagnose duplicated sample ownership
from per-rank evidence and repair it with a `DistributedSampler`.

A correct epoch has all three properties below:

- `processed_assignments` equals `dataset_size`;
- `unique_samples` equals `dataset_size`;
- `duplicate_assignments` is zero.

The resulting `exactly_once_global` field must therefore be `true`.

## Prerequisites

- basic familiarity with a PyTorch `Dataset` and `DataLoader`;
- the DDP setup introduced by the parent
  [`ddp_resnet50.py`](../ddp_resnet50.py) example;
- an environment containing PyTorch with distributed support;
- at least two `torchrun` processes. GPUs are useful but not required for this
  correctness exercise.

## Files

| File | Purpose |
|---|---|
| [`buggy_data_sharding.py`](buggy_data_sharding.py) | Starting point with an intentionally incorrect input pipeline. It reports the defect but exits successfully because the failure is the expected exercise result. |
| [`solution_data_sharding.py`](solution_data_sharding.py) | Complete reference implementation using `DistributedSampler` and `set_epoch()`. It exits unsuccessfully if the exactly-once check fails. |

The scripts are intentionally similar. A file comparison makes the small but
important input-pipeline changes visible; it is not necessary to rewrite DDP
boilerplate during the lab.

## Part 1: predict, run, and diagnose

Before running the buggy program, predict the answers to these questions:

1. With a dataset of 32 samples and two ranks, how many sample assignments
   should one correct global epoch contain?
2. Does wrapping the model in DDP also alter the `DataLoader`?
3. If every rank constructs the same ordinary shuffled `DataLoader`, which
   sample identifiers do you expect the ranks to share?

Run the starter program on CPU from this directory:

```bash
torchrun --standalone --nproc-per-node=2 buggy_data_sharding.py \
    --device cpu \
    --output buggy-report.json
```

On a GPU node, omit `--device cpu` or use `--device cuda`. With the defaults and
two ranks, the important output is:

```text
assignments: 64; unique samples: 32; duplicate assignments: 32
exactly once globally: False
```

Inspect the rank-specific sample-ID lists and the
`pairwise_rank_overlap` field in `buggy-report.json`. Explain why the model can
still train and remain synchronized even though this is not useful data
parallelism.

## Part 2: repair the input pipeline

Work in a personal copy of the buggy script. Change only the data-loading and
epoch setup needed to meet the success criterion. Do not use the reference
solution until you have a proposed diagnosis.

### Hints

1. DDP owns the model and gradient communication; look at
   `make_dataloader()` for the separate data-ownership decision.
2. A `DistributedSampler` needs the number of replicas and this process's
   global rank. When a sampler supplies indices, `DataLoader(shuffle=True)` is
   neither needed nor allowed.
3. For deterministic shuffling that changes between epochs, call
   `sampler.set_epoch(epoch)` before iterating over the loader.

Run your repaired version with the same arguments. A correct two-rank run
should report 32 assignments, 32 unique samples, zero duplicate assignments,
and `exactly once globally: True` in every epoch.

## Part 3: compare with the reference

Run the complete solution:

```bash
torchrun --standalone --nproc-per-node=2 solution_data_sharding.py \
    --device cpu \
    --output solution-report.json
```

Then compare the implementations:

```bash
diff -u buggy_data_sharding.py solution_data_sharding.py
```

The essential repair is small:

- `DistributedSampler` assigns a disjoint subset of indices to every rank;
- the loader receives `sampler=sampler` instead of `shuffle=True`;
- `sampler.set_epoch(epoch)` changes the sampler's deterministic shuffle seed
  for each epoch.

`set_epoch()` does not provide the sharding itself—the sampler does. Conversely,
the sampler can shard without `set_epoch()`, but its shuffled ordering then
repeats each epoch. As an extension, comment out `set_epoch()` in a copy of the
solution and compare the rank-specific order across epochs.

## Run as a Slurm batch job

For a single node with four assigned GPUs, one Slurm task can start four local
DDP processes:

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:05:00

srun torchrun --standalone --nproc-per-node=4 \
    solution_data_sharding.py \
    --dataset-size 32 \
    --output "solution-${SLURM_JOB_ID}.json"
```

Adapt the resource directives to the cluster; some sites require a GPU
partition, account, or module setup. The scripts use NCCL when CUDA is selected
and Gloo for the CPU fallback.

## Core code map

Function names are more durable than line numbers if the files change later.

| Area | Buggy starter | Reference solution | What to inspect |
|---|---|---|---|
| Indexed input | [`IndexedClassificationDataset` — line 47](buggy_data_sharding.py#L47) | [`IndexedClassificationDataset` — line 45](solution_data_sharding.py#L45) | Creates deterministic samples and returns a visible sample ID with each item. |
| Distributed setup | [`initialize_distributed()` — line 131](buggy_data_sharding.py#L131) | [`initialize_distributed()` — line 129](solution_data_sharding.py#L129) | Reads `torchrun` variables, chooses NCCL/CUDA or Gloo/CPU, and initializes the process group. |
| Model creation | [`make_model()` — line 194](buggy_data_sharding.py#L194) | [`make_model()` — line 192](solution_data_sharding.py#L192) | Builds the same tiny classifier on every rank. |
| Input handling | [`make_dataloader()` — line 204](buggy_data_sharding.py#L204) | [`make_dataloader()` — line 202](solution_data_sharding.py#L202) | Contains the deliberate defect and its sampler-based repair. |
| Training cycle | [`train_one_epoch()` — line 221](buggy_data_sharding.py#L221) | [`train_one_epoch()` — line 230](solution_data_sharding.py#L230) | Runs forward, backward, DDP gradient synchronization, and the optimizer step while recording sample IDs. |
| Evidence | [`analyze_assignments()` — line 273](buggy_data_sharding.py#L273) | [`analyze_assignments()` — line 282](solution_data_sharding.py#L282) | Computes coverage, duplication, missing IDs, and pairwise rank overlap. |
| Orchestration | [`run()` — line 338](buggy_data_sharding.py#L338) | [`run()` — line 347](solution_data_sharding.py#L347) | Connects distributed setup, input, model construction, training, and reporting. |

## Assumptions and limitations

- The dataset size must be divisible by the world size. In general,
  `DistributedSampler(drop_last=False)` pads its index list when divisibility is
  impossible, so a few repeated samples can be expected rather than erroneous.
  This exercise rejects that case to keep the exactly-once test unambiguous.
- Identical sample order is deliberately encouraged in the buggy script by
  using the same loader seed on every rank. Independent seeds would change the
  order but would **not** fix the ownership bug: every rank would still process
  the complete dataset.
- A changed permutation is random behavior, not a mathematical guarantee for
  tiny shards. The solution verifies exactly-once ownership and reports whether
  each observed ordering changed; the call to `set_epoch()` is the relevant
  implementation check.
- `gather_object()` is suitable for the small diagnostic lists used here, but
  object collectives are inefficient for large datasets. Production code
  should avoid gathering every sample ID during normal training and use compact
  counters or purpose-built validation instead.

## Further reading

- [PyTorch `DistributedDataParallel` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch `DistributedSampler` documentation](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
