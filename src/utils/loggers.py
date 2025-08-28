from pathlib import Path
import csv, time, json

class CSVLogger:
    def __init__(self, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(out_dir) / 'metrics.csv'
        self.file = open(self.path, 'w', newline='')
        self.writer = None

    def log(self, **kwargs):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=sorted(kwargs.keys()))
            self.writer.writeheader()
        self.writer.writerow(kwargs)
        self.file.flush()

    def close(self):
        self.file.close()

def save_config(cfg, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir)/'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
