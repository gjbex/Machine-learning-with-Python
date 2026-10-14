# Reference solution: empirical CUDA memory diagnosis

Run the commands in the exercise before reading this file. Exact MiB values are
hardware- and version-dependent; the phase and scaling patterns are the answer.

## Reference procedure

The following is a complete runnable collection and comparison sequence:

```bash
python memory_diagnostics.py --case alpha --output alpha.json
python memory_diagnostics.py --case beta --output beta.json
python memory_diagnostics.py --case gamma --output gamma.json

python compare_memory_reports.py alpha.json beta.json gamma.json

python memory_diagnostics.py --case alpha --batch-size-per-device 1 \
    --output alpha-batch1.json
python memory_diagnostics.py --case alpha --batch-size-per-device 4 \
    --output alpha-batch4.json

python memory_diagnostics.py --case beta --preset medium \
    --batch-size-per-device 1 --output beta-medium.json
python memory_diagnostics.py --case gamma --steps 10 \
    --output gamma-steps10.json
```

Skip the `medium` run if it is unsuitable for the allocated GPU. The three
default runs plus the alpha batch sweep are sufficient to meet the core
learning objective.

## Diagnosis

| Case | Dominant cause | Decisive evidence | Targeted response |
|---|---|---|---|
| `alpha` | Forward activations | Allocation rises during forward and its increment increases with local batch size, while the post-cleanup baseline remains stable across steps. | Reduce the microbatch, use activation checkpointing, or use a more memory-efficient kernel. |
| `beta` | AdamW optimizer state | The optimizer object initially adds almost no tensor storage, but a persistent allocation appears after the first `optimizer.step()` and remains after gradients are cleared. It grows with parameter count. | Shard model state, use a lower-memory optimizer, or reduce optimizer-state precision where scientifically acceptable. |
| `gamma` | Accidentally retained autograd graphs | `allocated_mib` after gradient clearing grows from one step to the next. More steps produce more growth even though model and batch dimensions are unchanged. | Stop retaining graphs and do not store graph-connected losses; retain only detached CPU values needed for logging. |

## Why the alternatives are not interchangeable

Activation checkpointing primarily removes selected forward intermediates and
recomputes them during backward. It is well targeted to `alpha`, but does not
remove AdamW's persistent moment tensors in `beta` and does not repair the
lifetime bug in `gamma`.

FSDP can shard parameters, gradients, and optimizer state, making it plausible
for `beta` when those states dominate. It may also reduce some model-related
baseline in the other cases, but it leaves ordinary unsharded activations local
and merely distributes a retained-graph bug across processes.

Calling `torch.cuda.empty_cache()` is not a repair. It can release unused cached
blocks to other applications, but it cannot free the live tensors represented
by `allocated_mib`. A large `reserved_mib - allocated_mib` gap is therefore not,
by itself, proof of a leak.

## Source confirmation

After making the empirical diagnosis, inspect `CASES` and `training_step()` in
[`memory_diagnostics.py`](memory_diagnostics.py):

- `alpha` uses stateless SGD and the normal graph lifetime;
- `beta` uses AdamW, whose moment buffers are initialized lazily at its first
  update;
- `gamma` requests `retain_graph=True` and keeps each graph-connected loss in a
  list.

The intentionally bad `gamma` behavior should be repaired in production as:

```python
loss.backward()
logged_losses.append(loss.detach().cpu().item())
```

Often it is better to maintain an online scalar sum rather than store every
value. Merely using `loss.detach()` while retaining the tensor on the GPU would
still keep that tensor's storage alive, although it would break the autograd
connection.

## Interpreting DDP evidence

With equal local workloads, ranks should have similar phase patterns, although
small differences can result from allocator and communication details. DDP can
also establish persistent gradient-bucket storage during the first backward,
so its first-step baseline need not equal the single-GPU result. With
`--rank-zero-extra-samples`, rank zero should show a larger forward increment.
The maximum across ranks is the relevant capacity constraint: the complete job
fails if any one rank exhausts its GPU.

PyTorch allocator counters omit some memory allocated directly by NCCL and
other CUDA libraries. A larger `device_used_mib` than `reserved_mib` is
therefore expected, but device-wide usage can also include unrelated processes
unless the GPU allocation is exclusive.
