import os
import sys
import time
import json
from pathlib import Path
from typing import Optional

# Make repo root importable so running this script directly works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config, apply_overrides, parse_cli
from src.utils.seed import set_seed
from src.utils.loggers import CSVLogger, save_config
from src.utils.storage import StorageManager
from src.models import build_model
from src.utils.cross_validation import RegularizerGridSearch, CrossValidator
from src.regularizers import build_regularizer
from src.evaluation.metrics import accuracy


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_dataloaders_from_cfg(cfg):
    """Return train_loader, test_loader or dataset objects depending on cross-validation setting."""
    ds_cfg = cfg.get('data', {})
    name = ds_cfg.get('dataset', 'cifar10')
    data_dir = ds_cfg.get('data_dir', './data')
    batch_size = ds_cfg.get('batch_size', 128)
    subset_ratio = ds_cfg.get('subset_ratio', None)
    test_subset_ratio = ds_cfg.get('test_subset_ratio', None)
    train_split = ds_cfg.get('train_split', 1.0)
    augment = ds_cfg.get('augment', False)

    # import dataset builders lazily to avoid heavy imports if not used
    if name == 'fashion_mnist':
        from src.datasets.fashion_mnist import build_fashion_mnist
        return build_fashion_mnist(data_dir, batch_size=batch_size, train_split=train_split,
                                    subset_ratio=subset_ratio, test_subset_ratio=test_subset_ratio)
    elif name == 'mnist':
        from src.datasets.mnist import build_mnist
        return build_mnist(data_dir, batch_size=batch_size, train_split=train_split,
                           subset_ratio=subset_ratio, test_subset_ratio=test_subset_ratio)
    elif name == 'cifar10':
        from src.datasets.cifar10 import build_cifar10
        return build_cifar10(data_dir, batch_size=batch_size, aug=augment,
                              subset_ratio=subset_ratio, test_subset_ratio=test_subset_ratio,
                              train_split=(train_split if train_split < 1.0 else None))
    else:
        raise ValueError(f'Unknown dataset: {name}')


def collect_activations(model, loader, device, layer_name: str):
    """Collect activations for all images in loader at the requested layer.

    Returns a numpy array with shape (num_images, feature_dim)
    """
    model.eval()
    acts_list = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb, collect_layer=layer_name)
            # model may return (logits, activations) or (None, activations)
            if isinstance(out, tuple) and len(out) == 2:
                _, acts = out
            else:
                acts = out

            if not acts:
                # nothing collected; try model.get_hidden or return empty
                if hasattr(model, 'get_hidden'):
                    feat = model.get_hidden(xb)
                    arr = feat.detach().cpu().numpy()
                else:
                    arr = np.zeros((xb.size(0), 0), dtype=np.float32)
            else:
                # pick the first available key
                if isinstance(acts, dict):
                    # choose requested layer if present
                    if layer_name in acts:
                        tensor = acts[layer_name]
                    else:
                        # fallback to first item
                        tensor = list(acts.values())[0]
                else:
                    tensor = acts
                tensor = tensor.detach().cpu()
                # flatten per-sample
                arr = tensor.view(tensor.size(0), -1).numpy()

            acts_list.append(arr)

    if len(acts_list) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(acts_list, axis=0)


def train_and_evaluate(cfg, reg_combo_name: str, reg_cfgs: list, run_dir: Path):
    device = get_device()
    train_loader, test_loader = build_dataloaders_from_cfg(cfg)

    model_cfg = cfg.get('model', {})
    model_name = model_cfg.get('name', 'simple_cnn')
    model_kwargs = model_cfg.get('kwargs', {})

    # Build model with regularizer parameters
    # instantiate regularizer objects and apply to model kwargs / model
    reg_objs = []
    for r in reg_cfgs:
        reg_name = r.get('name')
        reg_kwargs = r.get('kwargs', {})
        reg_objs.append(build_regularizer(reg_name, reg_kwargs))

    model = build_model(model_name, dataset=cfg.get('data', {}).get('dataset', 'cifar10'),
                        regularizers=reg_cfgs, **model_kwargs)

    # allow regularizers to mutate the model (e.g., set dropout p)
    for ro in reg_objs:
        try:
            model = ro.apply_to_model(model)
        except Exception:
            pass
    model = model.to(device)

    # Loss
    loss_cfg = cfg.get('loss', {})
    loss_name = loss_cfg.get('name', 'cross_entropy')
    if loss_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Unknown loss: {loss_name}')

    # Optimizer
    optim_cfg = cfg.get('optim', {})
    optim_name = optim_cfg.get('name', 'sgd')
    lr = optim_cfg.get('lr', 0.1)
    weight_decay = optim_cfg.get('weight_decay', 0.0)
    if optim_name == 'sgd':
        momentum = optim_cfg.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optim_name}')

    trainer = cfg.get('trainer', {})
    max_epochs = int(trainer.get('max_epochs', 10))

    # Loggers and storage
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(run_dir)
    save_config(cfg, run_dir)
    storage = StorageManager(base_dir=str(run_dir))

    # Which epoch(s) to collect activations? support 'final' or int
    analysis = cfg.get('analysis', {})
    collect_layer = analysis.get('layers', None)
    collect_epochs = analysis.get('collect_epochs', 'final')

    # training loop
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, yb)
            # add regularization penalties
            reg_pen = torch.tensor(0.0, device=loss.device)
            for ro in reg_objs:
                try:
                    p = ro.penalty(model)
                    if isinstance(p, torch.Tensor):
                        reg_pen = reg_pen + p.to(loss.device)
                except Exception:
                    pass
            loss = loss + reg_pen
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * yb.size(0)
            acc1 = accuracy(logits, yb, topk=(1,))[0]
            train_correct += acc1 * yb.size(0)
            train_total += yb.size(0)

        train_loss = train_loss / max(1, train_total)
        train_acc = float(train_correct / max(1, train_total))

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, yb)
                val_loss += loss.item() * yb.size(0)
                acc1 = accuracy(logits, yb, topk=(1,))[0]
                val_correct += acc1 * yb.size(0)
                val_total += yb.size(0)

        val_loss = val_loss / max(1, val_total)
        val_acc = float(val_correct / max(1, val_total))

        elapsed = time.time() - t0
        # write metrics
        csv_logger.log(epoch=epoch, time=elapsed, val_acc=val_acc, train_acc=train_acc,
                       val_loss=val_loss, train_loss=train_loss)

        # Possibly collect activations at this epoch
        should_collect = False
        if isinstance(collect_epochs, str) and collect_epochs == 'final' and epoch == max_epochs:
            should_collect = True
        elif isinstance(collect_epochs, int) and epoch == collect_epochs:
            should_collect = True

        if should_collect and collect_layer:
            try:
                acts = collect_activations(model, test_loader, device, collect_layer)
                acts_fname = analysis.get('activations_filename', 'activations.mat')
                # if extension is .mat or .h5, save HDF5; if .csv save CSV
                if acts_fname.endswith('.csv'):
                    # each row is an image vector
                    np.savetxt(run_dir / acts_fname, acts, delimiter=',')
                else:
                    # default: HDF5 with .mat extension
                    storage.save_activations({collect_layer: acts}, filename=acts_fname)
            except Exception as e:
                # be robust to collection failures
                print(f"Warning: failed to collect activations: {e}")

    csv_logger.close()


def expand_regularizers_and_run(cfg):
    out_base = cfg.get('trainer', {}).get('out_dir', 'runs')
    reg_cfgs = cfg.get('regularizers', [])

    # generate combinations using utility
    reg_search = RegularizerGridSearch(reg_cfgs)
    combinations = reg_search.generate_regularizer_combinations()

    # If cross-validation is enabled, we could run folds; for now create per-combo folder and run once.
    for combo_name, regs in combinations:
        run_dir = Path(out_base) / combo_name
        print(f"Running combo: {combo_name} -> out: {run_dir}")
        try:
            train_and_evaluate(cfg, combo_name, regs, run_dir)
        except Exception as e:
            print(f"Experiment {combo_name} failed: {e}")


def main():
    args = parse_cli()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    # seed
    seed = cfg.get('trainer', {}).get('seed', 42)
    set_seed(seed)

    # If visualization is enabled, leave a placeholder comment
    if cfg.get('visualization', {}).get('enabled', False):
        # Placeholder: visualization module will be implemented later
        # TODO: call visualization pipeline here when available
        pass

    # Expand regularizers and run experiments
    expand_regularizers_and_run(cfg)


if __name__ == '__main__':
    main()
