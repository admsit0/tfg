# TFG - A Study on Regularization Techniques Applied to Neural Networks

Author: Adam Maltoni (supervised by Dr. Luis Fernando Lago Fernández)

Overview
--------
This repository provides a modular experiment runner for testing regularization methods on small vision datasets (Fashion-MNIST, MNIST, CIFAR-10). The runner is configured via YAML files and supports:

- Flexible dataset builders (train/test splits, subsets, simple augmentation)
- Multiple model architectures (Simple feed-forward, simple CNN, ResNet18)
- Pluggable regularizers (dropout, L1; easy to extend via `src/regularizers/`)
- Cross-validation orchestration (k-fold) and per-fold outputs
- Activation collection (saved as HDF5/.mat or CSV; each row = flattened activation vector for an image)
- CSV logging of per-epoch metrics (`metrics.csv`) and saving the used config (`config.json`)

Entry point
-----------
Use `src/main.py` as the experiment runner. Example:

```powershell
python src/main.py --config configs/fashion_cv_dropout_l1.yaml
```

You can override configuration values on the CLI with the `--override` flag, using key=value pairs, e.g.: `--override trainer.max_epochs=5 data.subset_ratio=0.1`.

Directory layout (relevant files)
--------------------------------
- `src/main.py` — main experiment runner
- `src/datasets/` — dataset builder functions (FashionMNIST, MNIST, CIFAR-10)
- `src/models/` — model implementations and registry
- `src/regularizers/` — small, extensible regularizer interface and implementations
- `src/utils/` — config parsing, logging, seeding, cross-validation helpers, storage
- `configs/` — example YAML configs
- `runs/` — output directory (per-config / per-regularizer / per-fold)

Outputs produced by an experiment
-------------------------------
For each regularizer combination (and each fold when using cross-validation) the runner produces a directory like `runs/<combo>/fold1/` that contains:

- `metrics.csv` — per-epoch rows: epoch, time, val_acc, train_acc, val_loss, train_loss
- `config.json` — the config actually used for this run (including CLI overrides)
- activation file (name from config, e.g. `activations.mat` or `activations.csv`) — when activation collection is enabled, this file contains activations for the requested layer. Format:
	- HDF5 / .mat: datasets for the layer and `indices` (dataset indices mapping rows to original images)
	- CSV: rows are `[index, feat_0, feat_1, ...]`

Design notes
------------
- Regularizers expose two methods: `apply_to_model(model)` (to set dropout rates or similar) and `penalty(model)` (returns a scalar tensor to add to loss).
- Activation collection requests a layer name (for example `hidden` for the simple FFW). Models implement `collect_layer` semantics or expose `get_hidden()`.
- Seeding: the runner sets Python/numpy/torch seeds and uses per-fold deterministic generators for DataLoader shuffling. This improves reproducibility within the same environment.

CONFIG TUTORIAL
===============
Below is a detailed walkthrough of the YAML config format and every option currently supported. Use the file `configs/fashion_cv_dropout_l1.yaml` as the running example; the full example is shown and each option is explained below.

Example YAML (annotated)
------------------------
```yaml
data:
	dataset: fashion_mnist      # dataset name: 'fashion_mnist', 'mnist', or 'cifar10'
	data_dir: ./data           # path where datasets are downloaded/stored
	batch_size: 128            # batch size for training and validation loaders
	subset_ratio: 1            # optional fraction (0-1] of training set to use (for quick tests)
	test_subset_ratio: 1       # optional fraction of test/val set to use
	train_split: 1.0           # if <1.0, split the original train set into train/val using this fraction for training
	augment: false             # whether to use augmentation (used by CIFAR builder)

model:
	name: simple_ffw           # model identifier: 'simple_ffw', 'simple_cnn', 'resnet18'
	kwargs:
		input_dim: 784           # model constructor kwargs (example: input_dim for simple_ffw)
		hidden_dim: 256
		num_classes: 10

loss:
	name: cross_entropy        # loss to use; currently 'cross_entropy' supported
	kwargs: {}

optim:
	name: sgd                  # optimizer name: 'sgd' or 'adam'
	lr: 0.1                    # learning rate
	weight_decay: 0.0005
	momentum: 0.9              # only for SGD

trainer:
	max_epochs: 20             # number of training epochs
	out_dir: runs/fashion_cv_dropout_l1  # base output directory for runs
	seed: 42                   # global RNG seed used to initialize random, numpy, torch

cross_validation:
	enabled: true              # whether to run k-fold cross-validation
	n_folds: 5                 # number of folds (k)
	seed: 42                   # seed used by cross-validator for split reproducibility

regularizers:
	- name: dropout            # list of regularizers to apply; each regularizer can have kwargs
		kwargs: {p: 0.0}
		grid: [0.0, 0.2, 0.5]    # grid of values to expand into separate experiments
	- name: l1
		kwargs: {weight: 0.0}
		grid: [0.0, 1e-6, 1e-5, 1e-4]

analysis:
	collect_activations: true  # whether to collect activations
	layers: 'hidden'           # layer name to collect (model-dependent)
	quantize: null             # reserved (not implemented): quantization of activations
	collect_epochs: 'final'    # which epoch to collect activations ('final' or an int epoch)
	activations_filename: activations_collected.csv/.mat/.h5

visualization:
	enabled: false             # placeholder flag; no visualization pipeline is executed yet
	tqdm: null                 # set true to enable tqdm progress bars for runs
```

Detailed option explanations
----------------------------
- data.dataset (`str`)
	- 'fashion_mnist', 'mnist', or 'cifar10'. This selects which dataset builder to use.

- data.data_dir (`str`)
	- Path where datasets will be downloaded/stored.

- data.batch_size (`int`)
	- Batch size for training and validation.

- data.subset_ratio (`float` or null)
	- If set to a value in (0,1), randomly sample that fraction of the training set for fast experiments.

- data.test_subset_ratio (`float` or null)
	- Similar to subset_ratio but applied to the validation/test dataset.

- data.train_split (`float`)
	- If < 1.0, the training set is split into a smaller training set and a validation set. If =1.0, the official test set is used for evaluation.

- data.augment (`bool`)
	- If true, dataset builders that support augmentation (e.g., CIFAR) will apply simple augmentations (random crop, horizontal flip).

- model.name (`str`)
	- Model identifier. The constructor receives `model.kwargs` as parameters. See `src/models/*` for accepted kwargs per model.

- model.kwargs
	- Arbitrary keyword args forwarded to the model constructor. Typical fields:
		- `input_dim` (simple_ffw)
		- `hidden_dim` (simple_ffw)
		- `num_classes` (all models)
		- `dropout` or `in_channels` for convolutional models

- loss.name (`str`)
	- Currently only `cross_entropy` is implemented.

- optim.*
	- `name`: optimizer name (`sgd` or `adam`)
	- `lr`: learning rate
	- `weight_decay`: weight decay (L2) applied by the optimizer
	- `momentum`: SGD momentum (SGD only)

- trainer.*
	- `max_epochs`: training epochs
	- `out_dir`: root directory where `runs/` style outputs will be written
	- `seed`: global seed used by `src/utils/seed.set_seed`

- cross_validation.*
	- `enabled`: whether to run k-fold CV
	- `n_folds`: number of folds
	- `seed`: seed for generating folds

- regularizers (list)
	- Each element is a dict describing a regularizer to apply to the model. Supported fields:
		- `name`: regularizer name (e.g., `dropout`, `l1`)
		- `kwargs`: dictionary of parameters for this regularizer (e.g., `{p:0.2}` for dropout)
		- `grid`: optional list of values to expand the experiments across (the cross-validation helper will create one experiment per grid value)

- analysis.*
	- `collect_activations`: if true, the runner will collect activations at the layer named in `layers` for the epoch(s) specified.
	- `layers`: a layer name (string) or list of names; model implementations expose layer names they support (see `src/models/*`). Common values: `hidden`, `conv1`, `avgpool`.
	- `quantize`: reserved for future use.
	- `collect_epochs`: 'final' (collect after last epoch) or integer epoch number.
	- `activations_filename`: file name to save activations; extension `.csv` will create a CSV with indices as the first column; otherwise activations are saved in HDF5 (.mat) with both `indices` and the layer dataset.

- visualization.*
	- `enabled`: placeholder. When true, the runner prints a visible placeholder comment and can be extended to call visualization routines later.
	- `tqdm`: when true, per-fold and per-run progress bars are shown.

Practical notes and recommendations
----------------------------------
- Reproducibility: set `trainer.seed` and `cross_validation.seed` to fixed integers. Results are reproducible within the same environment (OS, Python, CUDA/cuDNN versions).
- Regularizers: use the `regularizers` list to try multiple techniques. The grid field expands experiments automatically (e.g., different dropout probabilities).
- Activations: collecting activations for cross-validated models can be done but be mindful of interpretation: different folds train different model instances; averaging activations across folds produces ensemble-like representations (useful, but document it). If you want stable activations per image, consider training a final model on the full training set and collecting activations from it.

Extending the code
------------------
- Add a new regularizer: create a new module in `src/regularizers/` implementing `apply_to_model` and `penalty`, then register it in the package registry.
- Add a new model: implement it in `src/models/` and register the constructor in `MODEL_REGISTRY`.
- Add visualization routines: implement plotting utilities and hook them into the `visualization.enabled` branch in `src/main.py`.

License & contact
-----------------
This is research code; please reach out to the author for questions or reuse requests.
