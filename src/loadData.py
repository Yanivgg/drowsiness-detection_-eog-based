"""
Data loading functions for microsleep detection
Adapted from the original project for:
- Binary classification (2 classes instead of 4)
- EOG-only signals (2 channels instead of 3 with EEG)
"""

import scipy.io as spio
import numpy as np


def load_recording(dir_name, f_name, n_cl=2):
    """
    Load a single .mat recording file
    
    Args:
        dir_name: directory containing the .mat file
        f_name: filename of the .mat file
        n_cl: number of classes (default 2 for binary)
        
    Returns:
        tuple: (E1, E2, EOG, targets_O1, targets_O2)
            - E1: ROC channel (right EOG)
            - E2: LOC channel (left EOG)
            - EOG: concatenated E1 and E2
            - targets_O1, targets_O2: labels
    """
    mat = spio.loadmat(dir_name + f_name, struct_as_record=False, squeeze_me=True)
    
    # Load labels
    labels_O1 = mat['Data'].labels_O1
    labels_O2 = mat['Data'].labels_O2
    
    # Load EOG channels (E1 = ROC, E2 = LOC)
    # Note: In the original project E1 and E2 are swapped in naming
    LOC = mat['Data'].E2  # Left EOG
    ROC = mat['Data'].E1  # Right EOG
    
    # Expand dimensions to match expected shape
    LOC = np.expand_dims(LOC, axis=-1)
    LOC = np.expand_dims(LOC, axis=-1)
    ROC = np.expand_dims(ROC, axis=-1)
    ROC = np.expand_dims(ROC, axis=-1)
    
    # Concatenate to create EOG with 2 channels
    EOG = np.concatenate((LOC, ROC), axis=2)
    
    targets_O1 = labels_O1
    targets_O2 = labels_O2
    
    return (ROC, LOC, EOG, targets_O1, targets_O2)


def classes(dir_name, f_name):
    """
    Get class labels from a recording
    
    Args:
        dir_name: directory containing the .mat file
        f_name: filename of the .mat file
        
    Returns:
        array: labels from the recording
    """
    mat = spio.loadmat(dir_name + f_name, struct_as_record=False, squeeze_me=True)
    labels_O1 = mat['Data'].labels_O1
    st = labels_O1
    return st


def classes_global(data_dir, files_train):
    """
    Get all labels from training files for class weight computation
    
    Args:
        data_dir: directory containing .mat files
        files_train: list of training filenames
        
    Returns:
        list: all labels concatenated from all training files
    """
    print("=====================")
    print("Reading train set for class distribution:")
    f_list = files_train
    st0 = []
    
    for i in range(len(f_list)):
        f = f_list[i]
        st = classes(data_dir, f)
        st0.extend(st)
    
    return st0


def load_data(data_dir, files_train, w_len=200*2):
    """
    Load multiple recordings with padding for sliding window
    
    Args:
        data_dir: directory containing .mat files
        files_train: list of filenames to load
        w_len: half window length in samples (default 200*2 = 400 samples)
        
    Returns:
        tuple: (data_train, targets_train, N_samples)
            - data_train: list of [E1, E2, EOG] for each file
            - targets_train: list of [targets1, targets2] for each file
            - N_samples: total number of samples across all files
    """
    E1, E2, X2, targets1, targets2 = load_recording(data_dir, files_train[0])
    print(E1.shape)
    
    data_train = []
    targets_train = []
    
    f_list = files_train
    
    # Padding for sliding windows
    padding = ((w_len, w_len), (0, 0), (0, 0))
    
    N_samples = 0
    for i in range(len(f_list)):
        f = f_list[i]
        print(f)
        E1, E2, X2, targets1, targets2 = load_recording(data_dir, f)
        
        # Apply padding
        E1 = np.pad(E1, pad_width=padding, mode='constant', constant_values=0)
        E2 = np.pad(E2, pad_width=padding, mode='constant', constant_values=0)
        X2 = np.pad(X2, pad_width=padding, mode='constant', constant_values=0)
        
        l = targets1.shape[0]
        N_samples += 2 * len(targets1)
        
        data_train.append([E1, E2, X2])
        targets_train.append([targets1, targets2])
    
    return (data_train, targets_train, N_samples)

