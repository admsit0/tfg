"""
Utility to generate multiple YAML experiment files by varying one or more fields.
Generates YAMLs and optional .bat/.sh runner scripts to execute them.
"""
import os
import yaml
from copy import deepcopy


def generate_variants(base_yaml_path, output_dir, param_name, values, prefix='exp'):
    """Generate variants of base_yaml changing param_name to each value in values.

    param_name supports nested keys separated by dots, e.g. 'regularizers.0.kwargs.p'
    """
    with open(base_yaml_path, 'r') as f:
        base = yaml.safe_load(f)
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    for i, v in enumerate(values):
        cfg = deepcopy(base)
        # set nested key
        parts = param_name.split('.')
        node = cfg
        for p in parts[:-1]:
            if p.isdigit():
                node = node[int(p)]
            else:
                node = node.setdefault(p, {})
        last = parts[-1]
        if last.isdigit():
            node[int(last)] = v
        else:
            node[last] = v
        out_name = f"{prefix}_{i}.yaml"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'w') as of:
            yaml.safe_dump(cfg, of)
        generated.append(out_path)
    return generated


def generate_run_script(yaml_paths, script_path, python_cmd='python'):
    """Generate a .bat or .sh script that runs all yaml experiments sequentially."""
    ext = os.path.splitext(script_path)[1]
    lines = []
    if ext.lower() == '.bat':
        for p in yaml_paths:
            lines.append(f"{python_cmd} src/train.py --config {p}")
    else:
        lines.append("#!/usr/bin/env bash")
        for p in yaml_paths:
            lines.append(f"{python_cmd} src/train.py --config {p}")
    with open(script_path, 'w') as sf:
        sf.write('\n'.join(lines))
    try:
        os.chmod(script_path, 0o755)
    except Exception:
        pass
    return script_path
