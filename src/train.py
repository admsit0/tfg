import os, time, json, sys
import torch
import torch.nn.functional as F
from torch import device, optim
from tqdm import tqdm
import multiprocessing
import logging
import csv
import numpy as np

# For Windows multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

if __package__ is None and __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.config import load_config, parse_cli, apply_overrides
from src.utils.seed import set_seed
from src.utils.loggers import CSVLogger, save_config
from src.datasets.cifar10 import build_cifar10
from src.models import build_model
from src.regularizers import build_regularizers, build_loss
from src.evaluation.metrics import accuracy
from src.analysis.hidden_state_stats import ActivationCollector
from src.utils.storage import StorageManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_optimizer(model, cfg):
    name = cfg.get('name','sgd').lower()
    lr = cfg.get('lr', 0.1)
    wd = cfg.get('weight_decay', 5e-4)
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f'Unknown optimizer: {name}')

def train_one_epoch(model, loader, optimizer, loss_fn, regs, device, collector=None, time_logger=None):
    logger.info("Starting training for one epoch...")
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    epoch_start_time = time.time()

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc='train', leave=False)):
        batch_start_time = time.time()
        logger.info(f"Processing batch {batch_idx + 1}/{len(loader)}")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        reg_term = torch.tensor(0.0, device=device)
        for r in regs:
            reg_term = reg_term + r.compute(model, (x, y), logits, loss)
        (loss + reg_term).backward()
        optimizer.step()

        with torch.no_grad():
            acc1 = accuracy(logits, y, topk=(1,))[0]
        total_loss += (loss + reg_term).item() * x.size(0)
        total_correct += acc1 * x.size(0)
        total_n += x.size(0)

        batch_time = time.time() - batch_start_time
        logger.info(f"Batch {batch_idx + 1} completed in {batch_time:.2f} seconds.")

        if time_logger:
            time_logger.writerow({"step": f"train_batch_{batch_idx + 1}", "time": batch_time})

    epoch_time = time.time() - epoch_start_time
    logger.info(f"Finished training for one epoch in {epoch_time:.2f} seconds.")
    if time_logger:
        time_logger.writerow({"step": "train_epoch", "time": epoch_time})
    return total_loss / total_n, total_correct / total_n

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, time_logger=None):
    logger.info("Starting evaluation...")
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    eval_start_time = time.time()

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc='eval', leave=False)):
        batch_start_time = time.time()
        logger.info(f"Evaluating batch {batch_idx + 1}/{len(loader)}")
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        acc1 = accuracy(logits, y, topk=(1,))[0]
        total_loss += loss.item() * x.size(0)
        total_correct += acc1 * x.size(0)
        total_n += x.size(0)

        batch_time = time.time() - batch_start_time
        logger.info(f"Batch {batch_idx + 1} evaluated in {batch_time:.2f} seconds.")

        if time_logger:
            time_logger.writerow({"step": f"eval_batch_{batch_idx + 1}", "time": batch_time})

    eval_time = time.time() - eval_start_time
    logger.info(f"Finished evaluation in {eval_time:.2f} seconds.")
    if time_logger:
        time_logger.writerow({"step": "eval_epoch", "time": eval_time})
    return total_loss / total_n, total_correct / total_n

def main():
    logger.info("Starting training process...")
    args = parse_cli()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    out_dir = cfg['trainer']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, out_dir)
    set_seed(cfg['trainer'].get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} (total {torch.cuda.device_count()} GPU(s))")
    else:
        logger.info("Using CPU")


    train_loader, test_loader = build_cifar10(
        cfg['data']['data_dir'],
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data'].get('num_workers', 0),
        aug=cfg['data'].get('augment', True),
        subset_ratio=cfg['data'].get('subset_ratio', None),
        test_subset_ratio=cfg['data'].get('test_subset_ratio', None),
        pin_memory=cfg['data'].get('pin_memory', True),
        train_split=cfg['data'].get('train_split', None),
        seed=cfg['trainer'].get('seed', 42)
    )



    model = build_model(cfg['model']['name'], **cfg['model'].get('kwargs', {})).to(device)

    regs = build_regularizers(cfg['regularizers'])
    for r in regs:
        r.setup(model)

    loss_cfg = cfg['loss']
    loss_fn = build_loss(loss_cfg.get('name','cross_entropy'), **loss_cfg.get('kwargs', {}))

    optimizer = build_optimizer(model, cfg['optim'])

    collector = None
    if cfg.get('analysis', {}).get('collect_activations', False):
        collector = ActivationCollector(
            model,
            module_prefixes=tuple(cfg['analysis'].get('module_prefixes',['features'])),
            quantize=cfg['analysis'].get('quantize', None)
        )

    # Rename the CSVLogger instance to avoid conflict with the global logger
    csv_logger = CSVLogger(out_dir)
    storage_manager = StorageManager(base_dir=out_dir)
    best_acc = 0.0

    # Open a CSV file to log time measurements
    time_log_path = os.path.join(out_dir, "time_log.csv")
    with open(time_log_path, "w", newline="") as time_log_file:
        time_logger = csv.DictWriter(time_log_file, fieldnames=["step", "time"])
        time_logger.writeheader()

        total_training_time = time.time()

        for epoch in range(cfg['trainer']['max_epochs']):
            logger.info(f"Starting epoch {epoch + 1}/{cfg['trainer']['max_epochs']}")
            epoch_start_time = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, regs, device, collector, time_logger)
            val_loss, val_acc = evaluate(model, test_loader, loss_fn, device, time_logger)

            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.")

            log_row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
            logger.info(f"Epoch {epoch + 1} results: {log_row}")
            logger.info("Saving metrics...")
            csv_logger.log(**log_row)

            if collector is not None and (epoch + 1) % cfg['analysis'].get('every', 5) == 0:
                logger.info("Aggregating activation statistics...")
                collector_start_time = time.time()
                stats = collector.aggregate()

                # Analyze activation distributions
                activation_distributions = {}
                for key, value in stats.items():
                    if isinstance(value, np.ndarray):
                        activation_distributions[key] = {
                            'mean': float(value.mean()),
                            'std': float(value.std()),
                            'histogram': np.histogram(value, bins=10)[0].tolist()
                        }
                    else:
                        activation_distributions[key] = value

                # Ensure all values in activation_distributions are JSON serializable (recursive handling)
                def make_serializable(obj):
                    if isinstance(obj, dict):
                        return {key: make_serializable(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [make_serializable(item) for item in obj]
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    return obj

                activation_distributions = make_serializable(activation_distributions)

                # Save activation distributions to JSON
                with open(os.path.join(out_dir, f'activation_distributions_epoch{epoch + 1}.json'), 'w') as f:
                    json.dump(activation_distributions, f)
                collector.storage.clear()
                collector_time = time.time() - collector_start_time
                logger.info(f"Activation statistics aggregated in {collector_time:.2f} seconds.")
                time_logger.writerow({"step": "activation_stats", "time": collector_time})

            if val_acc > best_acc:
                logger.info("New best accuracy achieved. Saving model...")
                torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(out_dir, 'best.pt'))

        total_training_time = time.time() - total_training_time
        logger.info(f"Total training process completed in {total_training_time:.2f} seconds.")
        logger.info("Saving final model...")
        torch.save({'model': model.state_dict(), 'epoch': cfg['trainer']['max_epochs'] - 1}, os.path.join(out_dir, 'last.pt'))
        logger.info("Training process completed.")

if __name__ == '__main__':
    main()
