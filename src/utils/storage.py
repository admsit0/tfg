import os
import h5py
import numpy as np
import json

class StorageManager:
    """
    A utility class for saving and loading data (e.g., activations, metrics) in an efficient format.
    """

    def __init__(self, base_dir="storage"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_activations(self, activations, filename):
        """
        Save activations to an HDF5 file.

        Args:
            activations (dict): A dictionary where keys are layer names and values are numpy arrays.
            filename (str): The name of the file to save the activations.
        """
        filepath = os.path.join(self.base_dir, filename)
        with h5py.File(filepath, "w") as f:
            for layer, data in activations.items():
                f.create_dataset(layer, data=data)

    def load_activations(self, filename):
        """
        Load activations from an HDF5 file.

        Args:
            filename (str): The name of the file to load the activations from.

        Returns:
            dict: A dictionary where keys are layer names and values are numpy arrays.
        """
        filepath = os.path.join(self.base_dir, filename)
        activations = {}
        with h5py.File(filepath, "r") as f:
            for layer in f.keys():
                activations[layer] = np.array(f[layer])
        return activations

    def save_metrics(self, metrics, filename):
        """
        Save metrics to a JSON file.

        Args:
            metrics (dict): A dictionary of metrics to save.
            filename (str): The name of the file to save the metrics.
        """
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, "w") as f:
            json.dump(metrics, f)

    def load_metrics(self, filename):
        """
        Load metrics from a JSON file.

        Args:
            filename (str): The name of the file to load the metrics from.

        Returns:
            dict: A dictionary of metrics.
        """
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, "r") as f:
            return json.load(f)
