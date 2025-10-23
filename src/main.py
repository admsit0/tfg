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
from src.visualization.visualization import run_visualization


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
        seed = cfg.get('trainer', {}).get('seed', 42)
        return build_fashion_mnist(data_dir, batch_size=batch_size, train_split=train_split,
                                    subset_ratio=subset_ratio, test_subset_ratio=test_subset_ratio, seed=seed)
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
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb, collect_layer=layer_name)
            # model may return (logits, activations) or (None, activations)
            if isinstance(out, tuple) and len(out) == 2:
                logits, acts = out
            else:
                # out may be dict of activations or logits
                if isinstance(out, dict):
                    logits = None
                    acts = out
                else:
                    logits = out
                    acts = None

            # determine features
            if not acts:
                if hasattr(model, 'get_hidden'):
                    feat = model.get_hidden(xb)
                    arr = feat.detach().cpu().numpy()
                else:
                    arr = np.zeros((xb.size(0), 0), dtype=np.float32)
            else:
                if isinstance(acts, dict):
                    if layer_name in acts:
                        tensor = acts[layer_name]
                    else:
                        tensor = list(acts.values())[0]
                else:
                    tensor = acts
                tensor = tensor.detach().cpu()
                arr = tensor.view(tensor.size(0), -1).numpy()

            # predictions and labels
            if logits is not None:
                if isinstance(logits, tuple):
                    logits = logits[0]
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            else:
                # no logits returned; mark preds as -1
                preds = -1 * np.ones((arr.shape[0],), dtype=np.int64)

            acts_list.append(arr)
            true_labels.append(yb.detach().cpu().numpy())
            pred_labels.append(preds)

    if len(acts_list) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    feats = np.concatenate(acts_list, axis=0)
    trues = np.concatenate(true_labels, axis=0)
    preds = np.concatenate(pred_labels, axis=0)
    return feats, trues, preds


def train_and_evaluate(cfg, reg_combo_name: str, reg_cfgs: list, run_dir: Path):
    raise RuntimeError('train_and_evaluate is deprecated; use train_and_evaluate_loaders')


def train_and_evaluate_loaders(cfg, reg_combo_name: str, reg_cfgs: list, run_dir: Path, train_loader, test_loader, reg_objs):
    device = get_device()

    model_cfg = cfg.get('model', {})
    # Reimplemented in train_and_evaluate_loaders
    # ...existing code...



def expand_regularizers_and_run(cfg):
    out_base = cfg.get('trainer', {}).get('out_dir', 'runs')
    reg_cfgs = cfg.get('regularizers', [])

    # generate combinations using utility
    reg_search = RegularizerGridSearch(reg_cfgs)
    combinations = reg_search.generate_regularizer_combinations()

    # If cross-validation is enabled, we could run folds; for now create per-combo folder and run once.
    # cross-validation settings
    cv_cfg = cfg.get('cross_validation', {})
    cv_enabled = cv_cfg.get('enabled', False)
    n_folds = int(cv_cfg.get('n_folds', 1)) if cv_enabled else 1
    cv_seed = int(cv_cfg.get('seed', 42))

    # tqdm option under visualization.tqdm
    viz_cfg = cfg.get('visualization', {})
    tqdm_enabled = bool(viz_cfg.get('tqdm', False))

    for combo_name, regs in combinations:
        run_dir = Path(out_base) / combo_name
        print(f"Running combo: {combo_name} -> out: {run_dir}")

        # build datasets and folds when requested
        if cv_enabled and n_folds > 1:
            # build full train dataset and split into folds
            from src.utils.cross_validation import CrossValidator
            train_dataset = None
            # build dataset but return underlying dataset by calling builders directly
            ds_cfg = cfg.get('data', {})
            name = ds_cfg.get('dataset', 'cifar10')
            data_dir = ds_cfg.get('data_dir', './data')
            batch_size = ds_cfg.get('batch_size', 128)
            # reuse builders to obtain dataset objects - we will request loaders per fold
            full_train_loader, _ = build_dataloaders_from_cfg(cfg)
            train_dataset = full_train_loader.dataset

            cv = CrossValidator(n_splits=n_folds, seed=cv_seed)
            folds = cv.split_dataset(train_dataset)

            # run each fold
            fold_results = []
            for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
                fold_name = f"fold{fold_idx}"
                fold_dir = run_dir / fold_name
                fold_dir.mkdir(parents=True, exist_ok=True)

                # construct loaders for this fold
                # derive per-fold base seed to ensure reproducible DataLoader order across folds
                fold_base_seed = int(cv_seed) + int(fold_idx)
                train_loader, val_loader = cv.create_fold_loaders(train_dataset, (train_idx, val_idx), batch_size=batch_size, base_seed=fold_base_seed)

                # reseed per-fold/model to ensure deterministic initialization independent of loop ordering
                import hashlib
                base_seed = int(cfg.get('trainer', {}).get('seed', 42))
                combo_hash = int(hashlib.md5(combo_name.encode()).hexdigest(), 16) % (2 ** 31 - 1)
                per_model_seed = (base_seed + combo_hash + fold_idx) % (2 ** 31 - 1)
                set_seed(per_model_seed)

                # build regularizer objects
                reg_objs = []
                for r in regs:
                    reg_name = r.get('name')
                    reg_kwargs = r.get('kwargs', {})
                    reg_objs.append(build_regularizer(reg_name, reg_kwargs))

                # Build model fresh per fold inside train function
                # show informative message
                if regs:
                    # show first reg setting
                    first = regs[0]
                    print(f"  Fold {fold_idx}: applying {first.get('name')} {first.get('kwargs')}")

                # optional tqdm wrapper
                if tqdm_enabled:
                    pbar = tqdm(total=int(cfg.get('trainer', {}).get('max_epochs', 10)), desc=f"{combo_name} fold{fold_idx}")
                else:
                    pbar = None

                # call per-fold train using loaders
                try:
                    # Build model and train using an internal helper adapted to accept loaders
                    # Prepare model and reg_objs inside helper
                    # reuse logic from train_and_evaluate but with loaders
                    # instantiate regularizers and model
                    model_cfg = cfg.get('model', {})
                    model_name = model_cfg.get('name', 'simple_cnn')
                    model_kwargs = model_cfg.get('kwargs', {})
                    model = build_model(model_name, dataset=cfg.get('data', {}).get('dataset', 'cifar10'), regularizers=regs, **model_kwargs)
                    for ro in reg_objs:
                        try:
                            model = ro.apply_to_model(model)
                        except Exception:
                            pass

                    # Now call a simplified training routine that supports per-epoch callbacks for tqdm
                    # Move training code inline here for clarity
                    device = get_device()
                    model = model.to(device)
                    # loss and optimizer
                    loss_cfg = cfg.get('loss', {})
                    criterion = nn.CrossEntropyLoss()
                    optim_cfg = cfg.get('optim', {})
                    optim_name = optim_cfg.get('name', 'sgd')
                    lr = optim_cfg.get('lr', 0.1)
                    weight_decay = optim_cfg.get('weight_decay', 0.0)
                    if optim_name == 'sgd':
                        momentum = optim_cfg.get('momentum', 0.9)
                        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                    max_epochs = int(cfg.get('trainer', {}).get('max_epochs', 10))
                    csv_logger = CSVLogger(fold_dir)
                    save_config(cfg, fold_dir)
                    storage = StorageManager(base_dir=str(fold_dir))
                    analysis = cfg.get('analysis', {})
                    collect_layer = analysis.get('layers', None)
                    collect_epochs = analysis.get('collect_epochs', 'final')

                    # training
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
                            # add reg penalties
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
                            for xb, yb in val_loader:
                                xb = xb.to(device)
                                yb = yb.to(device)
                                logits = model(xb)
                                if isinstance(logits, tuple):
                                    logits = logits[0]
                                loss_v = criterion(logits, yb)
                                val_loss += loss_v.item() * yb.size(0)
                                acc1 = accuracy(logits, yb, topk=(1,))[0]
                                val_correct += acc1 * yb.size(0)
                                val_total += yb.size(0)

                        val_loss = val_loss / max(1, val_total)
                        val_acc = float(val_correct / max(1, val_total))
                        elapsed = time.time() - t0

                        csv_logger.log(epoch=epoch, time=elapsed, val_acc=val_acc, train_acc=train_acc,
                                       val_loss=val_loss, train_loss=train_loss)

                        # tqdm update and info prints
                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix({'val_acc': f"{val_acc:.4f}", 'train_acc': f"{train_acc:.4f}"})
                        else:
                            print(f"    Epoch {epoch}/{max_epochs}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

                        # activation collection
                        should_collect = False
                        if isinstance(collect_epochs, str) and collect_epochs == 'final' and epoch == max_epochs:
                            should_collect = True
                        elif isinstance(collect_epochs, int) and epoch == collect_epochs:
                            should_collect = True
                        if should_collect and collect_layer:
                            try:
                                # For this trained model (fold_idx), collect activations for ALL folds' validation sets
                                acts_fname = analysis.get('activations_filename', 'activations_collected.csv')
                                save_per_epoch = bool(analysis.get('save_per_epoch', False))
                                suffix = Path(acts_fname).suffix or '.csv'
                                stem = Path(acts_fname).stem

                                # iterate all folds and save each fold's validation activations into this fold_dir
                                save_training = bool(analysis.get('save_training_activations', False))
                                for j, (t_idxs, v_idxs) in enumerate(folds, start=1):
                                    # build a deterministic DataLoader for this fold's validation subset
                                    # Only save activations for other folds if configured
                                    if j != fold_idx and not save_training:
                                        continue
                                    val_subset = torch.utils.data.Subset(train_dataset, v_idxs)
                                    def worker_init_fn_generic(worker_id, base= (int(cv_seed) + int(j)) % (2 ** 31 - 1)):
                                        seed = (int(base) + worker_id) % (2 ** 31 - 1)
                                        import random as _random, numpy as _np, torch as _torch
                                        _random.seed(seed)
                                        _np.random.seed(seed)
                                        try:
                                            _torch.manual_seed(seed)
                                        except Exception:
                                            pass

                                    val_loader_all = torch.utils.data.DataLoader(
                                        val_subset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn_generic
                                    )

                                    feats, trues, preds = collect_activations(model, val_loader_all, device, collect_layer)
                                    if feats.size == 0:
                                        combined = np.zeros((0, 0))
                                    else:
                                        combined = np.concatenate([feats, preds.reshape(-1, 1), trues.reshape(-1, 1)], axis=1)

                                    if save_per_epoch:
                                        out_name = f"{stem}_epoch{epoch}_fold{j}{suffix}"
                                    else:
                                        out_name = f"{stem}_fold{j}{suffix}"

                                    out_path = fold_dir / out_name
                                    # write header with activation columns + pred_label and true_label
                                    if combined.size == 0:
                                        # still write header
                                        header = 'pred_label,true_label'
                                        # create empty file with header only
                                        with open(out_path, 'w', newline='') as fh:
                                            fh.write(header + '\n')
                                    else:
                                        ncols = combined.shape[1]
                                        # last two are pred and true
                                        act_cols = [f"act_{i}" for i in range(ncols - 2)]
                                        header = ','.join(act_cols + ['pred_label', 'true_label'])
                                        # use fmt to avoid scientific notation issues if desired
                                        np.savetxt(out_path, combined, delimiter=',', header=header, comments='')

                            except Exception as e:
                                print(f"Warning: failed to collect activations across folds: {e}")

                    csv_logger.close()
                    if pbar is not None:
                        pbar.close()

                    fold_results.append((fold_idx, fold_dir))
                    print(f"  Fold {fold_idx} done; outputs in {fold_dir}")
                except Exception as e:
                    print(f"  Fold {fold_idx} failed: {e}")

            # after folds, produce a small summary
            print(f"Averaging results for combo {combo_name} over {len(fold_results)} folds")
            # (Optional) aggregate CSVs into a single summary per combo — left minimal for now

        else:
            # no cross-validation: single run
            try:
                # build loaders normally
                train_loader, test_loader = build_dataloaders_from_cfg(cfg)
                # prepare regularizers
                reg_objs = []
                for r in regs:
                    reg_objs.append(build_regularizer(r.get('name'), r.get('kwargs', {})))
                # reuse existing per-fold single-run training logic by wrapping loaders into one-fold run
                # create a fold-like directory
                fold_dir = run_dir / 'run'
                fold_dir.mkdir(parents=True, exist_ok=True)
                # call the same logic as above but for single run
                # build model and apply regularizers
                model_cfg = cfg.get('model', {})
                model_name = model_cfg.get('name', 'simple_cnn')
                model_kwargs = model_cfg.get('kwargs', {})
                model = build_model(model_name, dataset=cfg.get('data', {}).get('dataset', 'cifar10'), regularizers=regs, **model_kwargs)
                for ro in reg_objs:
                    try:
                        model = ro.apply_to_model(model)
                    except Exception:
                        pass
                # run training (call existing implementation via helper we created earlier)
                # For code reuse simplicity, call the earlier inline routine by constructing folds of size 1
                # Here we inline a call to the same per-fold training routine by creating small wrappers
                # (To avoid duplication, this could be refactored into a function.)
                # For now, call the same block as above by creating loaders variables
                # build optimizer, etc.
                device = get_device()
                model = model.to(device)
                loss_cfg = cfg.get('loss', {})
                criterion = nn.CrossEntropyLoss()
                optim_cfg = cfg.get('optim', {})
                optim_name = optim_cfg.get('name', 'sgd')
                lr = optim_cfg.get('lr', 0.1)
                weight_decay = optim_cfg.get('weight_decay', 0.0)
                if optim_name == 'sgd':
                    momentum = optim_cfg.get('momentum', 0.9)
                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                max_epochs = int(cfg.get('trainer', {}).get('max_epochs', 10))
                csv_logger = CSVLogger(fold_dir)
                save_config(cfg, fold_dir)
                storage = StorageManager(base_dir=str(fold_dir))
                analysis = cfg.get('analysis', {})
                collect_layer = analysis.get('layers', None)
                collect_epochs = analysis.get('collect_epochs', 'final')

                pbar = tqdm(total=max_epochs, desc=combo_name) if tqdm_enabled else None

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
                            loss_v = criterion(logits, yb)
                            val_loss += loss_v.item() * yb.size(0)
                            acc1 = accuracy(logits, yb, topk=(1,))[0]
                            val_correct += acc1 * yb.size(0)
                            val_total += yb.size(0)

                    val_loss = val_loss / max(1, val_total)
                    val_acc = float(val_correct / max(1, val_total))
                    elapsed = time.time() - t0
                    csv_logger.log(epoch=epoch, time=elapsed, val_acc=val_acc, train_acc=train_acc,
                                   val_loss=val_loss, train_loss=train_loss)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({'val_acc': f"{val_acc:.4f}", 'train_acc': f"{train_acc:.4f}"})
                    else:
                        print(f"  Epoch {epoch}/{max_epochs}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

                    should_collect = False
                    if isinstance(collect_epochs, str) and collect_epochs == 'final' and epoch == max_epochs:
                        should_collect = True
                    elif isinstance(collect_epochs, int) and epoch == collect_epochs:
                        should_collect = True
                        if should_collect and collect_layer:
                            try:
                                feats, trues, preds = collect_activations(model, test_loader, device, collect_layer)
                                if feats.size == 0:
                                    combined = np.zeros((0, 0))
                                else:
                                    combined = np.concatenate([feats, preds.reshape(-1, 1), trues.reshape(-1, 1)], axis=1)

                                acts_fname = analysis.get('activations_filename', 'activations_collected.csv')
                                save_per_epoch = bool(analysis.get('save_per_epoch', False))
                                suffix = Path(acts_fname).suffix or '.csv'
                                stem = Path(acts_fname).stem
                                if save_per_epoch:
                                    out_name = f"{stem}_epoch{epoch}_run{suffix}"
                                else:
                                    out_name = acts_fname

                                out_path = fold_dir / out_name
                                if combined.size == 0:
                                    header = 'pred_label,true_label'
                                    with open(out_path, 'w', newline='') as fh:
                                        fh.write(header + '\n')
                                else:
                                    ncols = combined.shape[1]
                                    act_cols = [f"act_{i}" for i in range(ncols - 2)]
                                    header = ','.join(act_cols + ['pred_label', 'true_label'])
                                    np.savetxt(out_path, combined, delimiter=',', header=header, comments='')
                            except Exception as e:
                                print(f"Warning: failed to collect activations: {e}")

                        # user requested not to save per-fold model checkpoints by default

                csv_logger.close()
                if pbar is not None:
                    pbar.close()
                print(f"Run complete; outputs in {fold_dir}")

            except Exception as e:
                print(f"Experiment {combo_name} failed: {e}")


def main():
    args = parse_cli()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    # seed
    seed = cfg.get('trainer', {}).get('seed', 42)
    set_seed(seed)

    # If visualization is enabled, call visualization runner
    viz_cfg = cfg.get('visualization', {})
    if viz_cfg.get('enabled', False):
        try:
            run_visualization(cfg)
        except Exception as e:
            print(f"Visualization runner failed: {e}")

    # Expand regularizers and run experiments
    expand_regularizers_and_run(cfg)


if __name__ == '__main__':
    main()
