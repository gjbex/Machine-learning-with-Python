# Machine Learning with Python redesign TODO

## Goal

Rebuild the training around a clear separation between machine-learning
methodology and deep-learning implementation:

- use scikit-learn to teach disciplined, reproducible machine-learning
  workflows;
- use native PyTorch to teach tensors, neural networks, optimization, and the
  training loop;
- introduce PyTorch Lightning only after participants understand the native
  PyTorch workflow;
- offer multi-GPU training as an optional advanced module rather than adding
  it to the four-hour core.

The redesign should preserve the current material until the replacement path
has been reviewed and validated.  Existing Keras examples should not be
removed merely because a PyTorch replacement has been drafted.

## Agreed framework responsibilities

### scikit-learn: methodology and classical machine learning

- [ ] Retain scikit-learn as the main framework for:
  - [ ] train, validation, and test-set discipline;
  - [ ] preprocessing pipelines and leakage prevention;
  - [ ] regression and classification baselines;
  - [ ] cross-validation and model selection;
  - [ ] evaluation metrics and diagnostic plots;
  - [ ] demonstrations of underfitting, overfitting, and inappropriate
        evaluation.
- [ ] Teach that a reproducible pipeline includes data, split definitions,
      fitted preprocessing, model configuration, software environment, and
      evaluation procedure; a random seed alone is insufficient.
- [ ] Retain clustering only if it supports a stated core learning outcome and
      fits within the four-hour schedule; otherwise move it to optional
      material.

### native PyTorch: deep-learning foundations

- [ ] Use native PyTorch to teach:
  - [ ] tensors, shapes, dtypes, and devices;
  - [ ] `Dataset`, transforms, and `DataLoader`;
  - [ ] `nn.Module` and the forward pass;
  - [ ] losses, automatic differentiation, and optimizers;
  - [ ] one explicit training loop and one explicit evaluation loop;
  - [ ] training, validation, and test modes;
  - [ ] saving and restoring a model and its relevant configuration;
  - [ ] CPU and single-GPU execution.
- [ ] Do not force image or sequence data through a scikit-learn `Pipeline`
      merely to preserve a uniform API.
- [ ] Explicitly map the methodological concepts from scikit-learn to their
      PyTorch mechanisms:

      | scikit-learn concept | PyTorch mechanism |
      |---|---|
      | `Pipeline` | `Dataset` plus composed transforms |
      | `train_test_split` | persisted split indices and `Subset` objects |
      | `random_state` | explicit generators and data-loader worker seeding |
      | estimator `fit()` | model, loss, optimizer, and training loop |
      | `predict()` and `score()` | evaluation loop and metric objects |
      | fitted preprocessing state | saved transform configuration and training statistics |

### PyTorch Lightning: optional orchestration layer

- [ ] Introduce Lightning only after the native PyTorch loop is understood.
- [ ] Present Lightning as an engineering trade-off: it organizes or
      automates training responsibilities but does not remove the need to
      understand them.
- [ ] Use Lightning for callbacks, checkpoints, logging, accelerator
      selection, and distributed strategies in the advanced material.
- [ ] Keep the native PyTorch example as the reference implementation when
      diagnosing unexpected Lightning behaviour.

### Related-training boundary

- [ ] Keep low-level GPU architecture, kernel programming, explicit
      host-device transfers, CuPy, Numba, PyCUDA, and RAPIDS in **Python on
      GPUs**.
- [ ] Let **Machine Learning with Python** own the machine-learning semantics
      of distributed training: data parallelism, global batch size, gradient
      synchronization, validation metrics, checkpoints, and convergence.
- [ ] Add reciprocal links between the two trainings instead of duplicating
      their core material.

## Priority 0: audit and protect the current teaching path

- [ ] Record which slides, notebooks, scripts, and exercises are part of the
      current live four-hour delivery and which are supplementary examples.
- [ ] Map every current learning outcome to the material and exercise that
      supports it.
- [ ] Identify which Keras examples must reach feature parity before the
      PyTorch path can replace them.
- [ ] Record expected runtimes and required hardware for all current neural
      network examples.
- [ ] Reconcile the documented four-hour duration with the current schedule,
      whose listed items exceed 240 minutes.
- [ ] Preserve the current Keras material on the development line until the
      replacement has passed the completion criteria below.
- [ ] Perform redesign work on a dedicated branch once implementation begins;
      avoid mixing the framework migration with unrelated documentation or
      environment changes.

## Priority 1: define the revised four-hour core

Use the following as a candidate schedule to validate, not as a final schedule
until a complete rehearsal has been timed:

| Topic | Time |
|---|---:|
| Introduction, research failure modes, and learning objectives | 15 min |
| Supervised-learning workflow and data splitting | 25 min |
| Scikit-learn pipelines, regression, and classification | 45 min |
| Evaluation, leakage, underfitting, and overfitting | 35 min |
| Break | 10 min |
| Neural-network concepts, tensors, and automatic differentiation | 25 min |
| PyTorch data path, model, and explicit training loop | 45 min |
| CNN hands-on exercise | 25 min |
| Reproducibility, limitations, and wrap-up | 15 min |
| **Total** | **240 min** |

- [ ] Decide whether regression and classification can share one coherent
      dataset and workflow without obscuring their differences.
- [ ] Use simple classical models as meaningful baselines before introducing a
      neural network.
- [ ] Ensure that preprocessing is fitted only on training data.
- [ ] Keep the test set untouched until the final evaluation.
- [ ] Require participants to compare training and validation behaviour, not
      only report one final accuracy value.
- [ ] Move recurrent networks, word embeddings, clustering, extensive
      hyperparameter optimization, and multi-GPU training out of the core
      unless rehearsal shows that they can be taught without rushing the
      methodological material.
- [ ] Update the learning outcomes before rewriting the slides so that every
      core topic has a reason to remain.

## Priority 2: establish the native PyTorch reference implementation

- [ ] Review and modernize `source-code/pytorch/tensors.ipynb` as the concise
      tensor and device introduction.
- [ ] Review and modernize `source-code/pytorch/mnist.ipynb` as the native
      training-loop reference.
- [ ] Refactor reusable code out of notebooks where doing so makes behaviour
      easier to test and reuse.
- [ ] Provide a small script version of the complete training workflow; do not
      make notebooks the only executable form.
- [ ] Give the model a normal `forward()` method and call the model rather than
      reaching into internal layers during evaluation.
- [ ] Use class-index targets with `CrossEntropyLoss` unless an example has a
      genuine need for one-hot encoded targets.
- [ ] Separate training transforms from validation and test transforms.
- [ ] Make shuffling behaviour explicit for each data loader.
- [ ] Persist or deterministically reconstruct the data split so that
      scikit-learn and PyTorch comparisons use the same observations.
- [ ] Add a tiny CPU smoke configuration that completes quickly.
- [ ] Add capability-based accelerator selection with a clear CPU fallback.
- [ ] Ensure the CPU fallback preserves scientific meaning rather than silently
      changing precision, preprocessing, or evaluation.

## Priority 3: carry methodological discipline into PyTorch

- [ ] Add a short comparison showing that the same leakage can occur in either
      scikit-learn or PyTorch when preprocessing uses validation or test data.
- [ ] Compute normalization statistics from the training subset only.
- [ ] Demonstrate distinct training and inference behaviour for dropout and
      batch normalization where relevant.
- [ ] Plot training and validation loss from recorded epoch-level metrics.
- [ ] Add a deliberately over-parameterized model and diagnose overfitting
      before applying regularization or early stopping.
- [ ] Compare the neural network against a credible scikit-learn baseline on
      the same split and metric.
- [ ] Explain why neural-network cross-validation and exhaustive grid search
      may be computationally inappropriate even though they are technically
      possible.
- [ ] Record model configuration, optimizer, learning rate, batch size, epoch
      count, split identity, and package versions with each run.
- [ ] Distinguish repeatable splitting, controlled randomness, deterministic
      operations, and reproducibility across platforms or accelerator types.
- [ ] Do not promise bitwise-identical CPU and GPU results.

## Priority 4: revise hands-on material

- [ ] Retain separate starter and completed versions of core exercises.
- [ ] Create one integrated exercise that follows this path:

      ```text
      inspect data
      -> define and preserve a split
      -> fit preprocessing on training data
      -> establish a classical baseline
      -> train a neural network
      -> diagnose training and validation behaviour
      -> evaluate once on the test data
      -> save the model, preprocessing information, and run configuration
      ```

- [ ] Ensure participants make at least one decision rather than only execute
      prepared cells.
- [ ] Include one exercise in which a plausible but methodologically invalid
      workflow produces an over-optimistic result.
- [ ] Keep datasets small enough for the CPU fallback, while documenting that
      GPU performance conclusions require a representative workload.
- [ ] Avoid using MNIST performance as evidence that multiple GPUs improve
      training speed.
- [ ] Add expected runtimes for CPU and the tested GPU configuration.

## Priority 5: introduce Lightning after native PyTorch

- [ ] Refactor the validated native PyTorch example into a
      `LightningModule` without changing the model or split.
- [ ] Compare the native and Lightning versions responsibility by
      responsibility: device placement, loops, metrics, callbacks, logging,
      checkpointing, and distributed sampling.
- [ ] Modernize `source-code/pytorch-lightning/mnist.ipynb` or replace it with
      a script plus a short analysis notebook.
- [ ] Add an explicit accelerator and device configuration rather than relying
      on unexplained auto-detection in the teaching example.
- [ ] Ensure validation and test metrics are reduced correctly across devices.
- [ ] Ensure only the intended rank writes checkpoints and other shared
      artifacts.
- [ ] Verify that early stopping monitors a correctly aggregated epoch-level
      validation metric.
- [ ] Test restoring a checkpoint on CPU and on a single GPU.
- [ ] Keep Lightning out of the mandatory core if it prevents participants
      from understanding the native training loop.

## Optional two-hour module: multi-GPU deep learning

Use the following as a candidate schedule:

| Topic | Time |
|---|---:|
| Data parallelism, ranks, world size, and collectives | 15 min |
| Establish and measure the single-GPU baseline | 15 min |
| Native PyTorch DDP anatomy | 15 min |
| Refactor or configure the Lightning implementation | 20 min |
| Break | 10 min |
| One-, two-, and four-GPU scaling experiment | 25 min |
| Correctness, reproducibility, metrics, and checkpoints | 15 min |
| Slurm/multi-node overview and conclusions | 5 min |
| **Total** | **120 min** |

- [ ] State explicitly that the first module covers synchronous data-parallel
      training, not model parallelism or distributed inference.
- [ ] Explain that data-parallel training replicates the model; it does not
      combine GPU memory into one larger memory pool.
- [ ] Teach one process per GPU, local and global ranks, world size, and
      gradient all-reduce before showing the Lightning configuration.
- [ ] Convert the distributed training example to a script; use notebooks for
      explanation and result analysis rather than as the primary DDP launcher.
- [ ] Provide a prepared Slurm submission script whose resource requests match
      the Lightning or PyTorch device configuration.
- [ ] Use MNIST only as a correctness and communication-overhead smoke test.
- [ ] Select a second, compute-intensive but operationally manageable workload
      for the scaling exercise.
- [ ] Compare fixed global batch size and fixed per-device batch size, and
      explain that they answer different scaling questions.
- [ ] Record throughput, epoch time, scaling efficiency, and validation
      behaviour; do not report speedup without checking model quality.
- [ ] Synchronize accelerator timing and separate data-loading, host-device
      transfer, computation, and communication costs where practical.
- [ ] Record GPU model, count, interconnect, node count, driver, PyTorch,
      Lightning, communication backend, precision, and input shape with every
      benchmark.
- [ ] Demonstrate at least one case where adding GPUs does not improve runtime
      and explain why.
- [ ] Treat multi-node execution as an overview or follow-on exercise unless
      participants have reliable access to multiple reserved nodes.
- [ ] Document CUDA/NCCL as the primary tested backend if that is what the
      training infrastructure provides; list ROCm or other backends as
      untested until they have actually been exercised.

## Optional and follow-on modules

- [ ] Decide which displaced topics justify independent modules rather than a
      collection of disconnected examples:
  - [ ] clustering and dimensionality reduction;
  - [ ] modern computer-vision workflows and transfer learning;
  - [ ] sequence models and transformers;
  - [ ] hyperparameter optimization and experiment tracking;
  - [ ] explainability and failure analysis;
  - [ ] distributed inference;
  - [ ] model, tensor, and fully sharded data parallelism.
- [ ] Give each optional module explicit prerequisites and learning outcomes.
- [ ] Avoid teaching framework catalogues; select one representative tool for
      each learning objective and explain its trade-offs.

## Keras migration and legacy material

- [ ] Maintain a mapping from every core Keras slide and exercise to its
      PyTorch replacement.
- [ ] Do not remove the Keras path until the PyTorch path covers the agreed
      learning outcomes and has been rehearsed successfully.
- [ ] Decide whether Keras examples should remain as:
  - [ ] an explicitly labelled legacy path;
  - [ ] optional framework-comparison material;
  - [ ] a separate archived branch or release;
  - [ ] removed material after an agreed deprecation period.
- [ ] Remove Keras and TensorFlow from the default environment only after no
      core material requires them.
- [ ] Avoid maintaining equivalent core exercises in both Keras and PyTorch
      indefinitely; duplicated teaching paths will drift.

## Documentation, environments, and quality assurance

- [ ] Update `docs/README.md`, `README.md`, `training.toml`, slides, and topic
      READMEs together when the revised scope is accepted.
- [ ] Correct the schedule metadata so its item durations sum to the declared
      course duration.
- [ ] Create a minimal core environment and isolate optional Lightning,
      distributed, or specialist dependencies where practical.
- [ ] Pin or lock a trainer-validated environment for each release of the
      course.
- [ ] Add a repeatable clean-kernel execution check for core notebooks.
- [ ] Add syntax and small-data smoke checks for training scripts.
- [ ] Run the core workflow on CPU in routine validation.
- [ ] Run a single-GPU smoke test on actual accelerator hardware before each
      delivery that advertises GPU support.
- [ ] Run multi-GPU validation only on real multi-GPU hardware; do not treat
      import success or mocked device counts as validation.
- [ ] Compare CPU and accelerator outputs using justified tolerances rather
      than assuming bitwise equality.
- [ ] Check that participant-facing notebooks contain no stale exceptions,
      machine-specific paths, credentials, or large generated artifacts.
- [ ] Perform a final slide-to-exercise-to-environment consistency check before
      publishing a revised release.

## Completion criteria

- [ ] The mandatory schedule totals exactly 240 minutes including its break.
- [ ] Every mandatory learning outcome has participant-facing material and at
      least one exercise or decision point.
- [ ] Scikit-learn remains the methodological anchor for classical workflows.
- [ ] Participants can explain how the same split, leakage, validation, and
      reproducibility principles apply in PyTorch.
- [ ] Participants can read and modify a native PyTorch training loop before
      encountering Lightning.
- [ ] All core examples run from a clean environment on CPU.
- [ ] Advertised GPU paths have been tested on documented hardware and runtime
      versions.
- [ ] The multi-GPU module reports both correctness and scaling results for
      one, two, and four GPUs, or clearly documents unavailable configurations.
- [ ] The Keras path is retained, labelled optional, or retired according to an
      explicit migration decision rather than removed incidentally.
- [ ] Repository documentation, slides, exercises, environments, and training
      metadata describe the same delivered course.
