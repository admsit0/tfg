import yaml, argparse, os, copy

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to YAML config')
    ap.add_argument('--override', type=str, nargs='*', default=None,
                    help="Override config as KEY=VALUE pairs, e.g., trainer.max_epochs=5 optim.lr=0.01")
    return ap.parse_args()

def deep_set(d, key, value):
    keys = key.split('.')
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    # try to cast types
    if isinstance(value, str):
        if value.lower() in ['true','false']:
            value = value.lower() == 'true'
        else:
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                pass
    cur[keys[-1]] = value

def apply_overrides(cfg, overrides):
    if not overrides: return cfg
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if '=' not in ov:
            continue
        k,v = ov.split('=',1)
        deep_set(cfg, k, v)
    return cfg
