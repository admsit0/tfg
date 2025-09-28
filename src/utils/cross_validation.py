"""
Cross-validation utilities for regularization method comparison.
"""
import torch
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Any, Tuple
import logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

class CrossValidator:
    """Handles K-fold cross-validation for model evaluation."""
    
    def __init__(self, n_splits: int = 5, seed: int = 42, shuffle: bool = True):
        self.n_splits = n_splits
        self.seed = seed
        self.shuffle = shuffle
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    
    def split_dataset(self, dataset):
        """Split dataset into k folds."""
        indices = list(range(len(dataset)))
        folds = []
        
        for train_idx, val_idx in self.kfold.split(indices):
            train_indices = [indices[i] for i in train_idx]
            val_indices = [indices[i] for i in val_idx]
            folds.append((train_indices, val_indices))
            
        return folds
    
    def create_fold_loaders(self, dataset, fold_indices, batch_size=128):
        """Create data loaders for a specific fold."""
        train_indices, val_indices = fold_indices
        
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader

class RegularizerGridSearch:
    """Handles grid search over regularization methods."""
    
    def __init__(self, regularizers_config: List[Dict], default_grids: Dict = None):
        self.regularizers_config = regularizers_config
        self.default_grids = default_grids or {}
        
    def generate_regularizer_combinations(self) -> List[Tuple[str, Dict]]:
        """Generate all combinations of regularizers with their parameter grids."""
        combinations = []
        
        for reg_config in self.regularizers_config:
            reg_name = reg_config['name']
            reg_kwargs = reg_config.get('kwargs', {})
            custom_grid = reg_config.get('grid', None)
            
            # Get parameter grid from the regularizers package
            from src.regularizers import get_regularizer_grid
            param_grid = get_regularizer_grid(reg_name, custom_grid)
            
            if not param_grid:
                # No grid, use single configuration
                combinations.append((f"{reg_name}", [{
                    'name': reg_name,
                    'kwargs': reg_kwargs
                }]))
            else:
                # Create combinations for each parameter value
                param_key = self._get_param_key(reg_name)
                for param_value in param_grid:
                    updated_kwargs = reg_kwargs.copy()
                    updated_kwargs[param_key] = param_value
                    
                    combination_name = f"{reg_name}_{param_key}_{param_value}"
                    combinations.append((combination_name, [{
                        'name': reg_name,
                        'kwargs': updated_kwargs
                    }]))
        
        # Add baseline (no regularization)
        combinations.append(("baseline", []))
        
        return combinations
    
    def _get_param_key(self, reg_name: str) -> str:
        """Get the main parameter key for each regularizer type."""
        param_mapping = {
            'dropout': 'p',
            'l1': 'weight',
            'l2': 'weight',
            'gaussian_proximity': 'lambda_reg'
        }
        return param_mapping.get(reg_name, 'weight')

