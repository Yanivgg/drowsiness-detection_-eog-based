"""
Utility functions for microsleep detection
Adapted from the original project
"""

import os
import numpy as np
from sklearn.metrics import cohen_kappa_score


def create_tmp_dirs(dirs):
    """
    Create directories if they don't exist
    
    Args:
        dirs: list of directory paths to create
    """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")


def batch_generator(sample_list, batch_size):
    """
    Generate batches from a list of samples
    
    Args:
        sample_list: list of samples
        batch_size: size of each batch
        
    Yields:
        batch: list of samples in current batch
    """
    for i in range(0, len(sample_list), batch_size):
        yield sample_list[i:i + batch_size]


def kappa_metric(y_true, y_pred, n_classes):
    """
    Calculate Cohen's kappa per class
    
    Args:
        y_true: true labels (1D array)
        y_pred: predicted labels (1D array)
        n_classes: number of classes
        
    Returns:
        kappa_per_class: array of kappa values per class
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    kappa_per_class = np.zeros(n_classes)
    
    for cl in range(n_classes):
        # Create binary problem for this class vs rest
        y_true_binary = (y_true == cl).astype(int)
        y_pred_binary = (y_pred == cl).astype(int)
        
        try:
            kappa_per_class[cl] = cohen_kappa_score(y_true_binary, y_pred_binary)
        except:
            kappa_per_class[cl] = 0.0
    
    return kappa_per_class

