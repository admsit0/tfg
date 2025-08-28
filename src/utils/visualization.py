import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
import json

class Visualization:
    """
    A utility class for visualizing activations, metrics, and other data.
    """

    def __init__(self, base_dir="storage"):
        self.base_dir = base_dir

    def plot_metrics(self, metrics_file, save_path=None):
        """
        Plot training and validation metrics over epochs.

        Args:
            metrics_file (str): Path to the JSON file containing metrics.
            save_path (str): Path to save the plot. If None, the plot is shown.
        """
        filepath = os.path.join(self.base_dir, metrics_file)
        with open(filepath, "r") as f:
            metrics = json.load(f)

        epochs = [entry['epoch'] for entry in metrics]
        train_loss = [entry['train_loss'] for entry in metrics]
        val_loss = [entry['val_loss'] for entry in metrics]
        train_acc = [entry['train_acc'] for entry in metrics]
        val_acc = [entry['val_acc'] for entry in metrics]

        plt.figure(figsize=(12, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_activation_distribution(self, activations_file, layer_name, save_path=None):
        """
        Plot the distribution of activations for a specific layer.

        Args:
            activations_file (str): Path to the HDF5 file containing activations.
            layer_name (str): Name of the layer to visualize.
            save_path (str): Path to save the plot. If None, the plot is shown.
        """
        filepath = os.path.join(self.base_dir, activations_file)
        with h5py.File(filepath, "r") as f:
            if layer_name not in f:
                raise ValueError(f"Layer {layer_name} not found in activations file.")
            activations = np.array(f[layer_name])

        plt.figure(figsize=(8, 6))
        sns.histplot(activations.flatten(), bins=50, kde=True)
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.title(f"Activation Distribution for Layer: {layer_name}")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def compare_activation_states(self, activations_file, layer_name, save_path=None):
        """
        Compare the number of unique activation states in a layer.

        Args:
            activations_file (str): Path to the HDF5 file containing activations.
            layer_name (str): Name of the layer to analyze.
            save_path (str): Path to save the plot. If None, the plot is shown.
        """
        filepath = os.path.join(self.base_dir, activations_file)
        with h5py.File(filepath, "r") as f:
            if layer_name not in f:
                raise ValueError(f"Layer {layer_name} not found in activations file.")
            activations = np.array(f[layer_name])

        unique_states = np.unique(activations, axis=0)
        num_unique_states = unique_states.shape[0]

        plt.figure(figsize=(8, 6))
        sns.histplot([len(state) for state in unique_states], bins=50, kde=False)
        plt.xlabel("Number of Unique States")
        plt.ylabel("Frequency")
        plt.title(f"Unique Activation States for Layer: {layer_name}\nTotal Unique States: {num_unique_states}")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
