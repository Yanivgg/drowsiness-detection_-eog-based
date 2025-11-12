"""
Feature Extraction Pipeline for EOG-based Drowsiness Detection
==============================================================

This script extracts comprehensive features from preprocessed .mat files
(created in Phase 1 for CNN training) using a sliding window approach.

WHY USE .MAT FILES?
- Already preprocessed (filtered, resampled to 200 Hz)
- Same preprocessing as Phase 1 CNN (ensures fair comparison)
- Labels already extracted from annotations
- Faster and more consistent

Configuration:
- Window: 16 seconds (3200 samples @ 200 Hz) - matches CNN_16s
- Stride: 1600 samples (8 seconds, 50% overlap) - balanced performance
- Train: Subjects 01-06, 08-10 (18 files)
- Test: Subject 07 (2 files)

Features extracted: ~61 total
- Time-domain: 26 features
- Frequency-domain: 16 features  
- Non-linear: 7 features (Sample/Approx Entropy removed - too slow)
- EOG-specific: 12 features (SEM features removed - always 0)

Performance: ~0.03s per window, ~10 minutes total runtime

Author: Yaniv Grosberg
Date: 2025-11-09
Updated: 2025-11-11 (removed SEM features)
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as spio
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering.time_domain import extract_time_domain_features
from feature_engineering.frequency_domain import extract_frequency_domain_features
from feature_engineering.nonlinear import extract_nonlinear_features
from feature_engineering.eog_specific import extract_eog_specific_features


# ============================================================
# CONFIGURATION
# ============================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'files')  # MAT files directory
FILE_SETS_PATH = os.path.join(BASE_DIR, 'data', 'file_sets.mat')
OUTPUT_DIR = os.path.join(BASE_DIR, 'phase_2_project', 'features')

# Window parameters
WINDOW_SIZE = 16  # seconds
WINDOW_SAMPLES = 3200  # samples (16s * 200 Hz)
STRIDE = 200  # samples (8 second stride = 50% overlap)
FS = 200  # sampling frequency

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_file_sets():
    """Load train/test split from file_sets.mat"""
    print("Loading file sets...")
    file_sets = spio.loadmat(FILE_SETS_PATH)
    
    # Phase 2: Train = 01-06, 08-10; Test = 07
    files_train = [f[0] for f in file_sets['files_train'].flatten()]
    files_val = [f[0] for f in file_sets['files_val'].flatten()]
    files_test = [f[0] for f in file_sets['files_test'].flatten()]
    
    # Combine train and val for Phase 2 (no validation set)
    files_train_phase2 = files_train + files_val
    files_test_phase2 = files_test
    
    print(f"[OK] Train files (Phase 2): {len(files_train_phase2)} files")
    print(f"     Subjects: 01-06, 08-10")
    print(f"[OK] Test files (Phase 2): {len(files_test_phase2)} files")
    print(f"     Subject: 07")
    
    return files_train_phase2, files_test_phase2


def load_mat_file(filepath):
    """
    Load a single .mat file (already preprocessed for Phase 1 CNN).
    
    These files already contain:
    - E1, E2: LOC and ROC channels resampled to 200 Hz
    - labels_O1, labels_O2: Binary labels (0=awake, 1=drowsy)
    
    Returns: sig_loc, sig_roc, labels, fs
    """
    try:
        # Load MAT file
        mat_data = spio.loadmat(filepath)
        
        # Extract Data structure
        if 'Data' not in mat_data:
            print(f"  Warning: 'Data' field not found in {filepath}")
            return None, None, None, None
        
        data = mat_data['Data']
        
        # Extract EOG channels
        # MAT structure: Data.E1, Data.E2, Data.labels_O1, Data.labels_O2
        sig_loc = data['E1'][0, 0].squeeze()  # LOC channel
        sig_roc = data['E2'][0, 0].squeeze()  # ROC channel
        labels = data['labels_O1'][0, 0].squeeze()  # Labels (same for O1 and O2)
        
        # Verify data
        if len(sig_loc) != len(sig_roc) or len(sig_loc) != len(labels):
            print(f"  Warning: Signal and label lengths don't match in {filepath}")
            return None, None, None, None
        
        return sig_loc, sig_roc, labels, FS
        
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None, None, None, None


def create_sliding_windows(signal_length, window_size, stride):
    """
    Create sliding window indices.
    
    Parameters
    ----------
    signal_length : int
        Total length of signal
    window_size : int
        Window size in samples
    stride : int
        Stride in samples
        
    Returns
    -------
    windows : list of tuples
        List of (start_idx, end_idx) tuples
    """
    windows = []
    for start in range(0, signal_length - window_size + 1, stride):
        end = start + window_size
        windows.append((start, end))
    return windows


def extract_window_features(loc_window, roc_window, fs=200):
    """
    Extract all features from a single window.
    
    Parameters
    ----------
    loc_window : np.ndarray
        LOC signal for this window
    roc_window : np.ndarray
        ROC signal for this window
    fs : float
        Sampling frequency
        
    Returns
    -------
    features : dict
        Dictionary of all extracted features
    """
    features = {}
    
    # Differential signal (horizontal EOG)
    diff_signal = loc_window - roc_window
    
    # 1. Time-domain features (on differential signal)
    td_features = extract_time_domain_features(diff_signal)
    features.update(td_features)
    
    # 2. Frequency-domain features (on differential signal)
    fd_features = extract_frequency_domain_features(diff_signal, fs)
    features.update(fd_features)
    
    # 3. Non-linear features (on differential signal)
    nl_features = extract_nonlinear_features(diff_signal)
    features.update(nl_features)
    
    # 4. EOG-specific features (using both LOC and ROC)
    eog_features = extract_eog_specific_features(loc_window, roc_window, fs)
    features.update(eog_features)
    
    return features


def assign_window_label(labels_in_window, threshold=0.1):
    """
    Assign label to a window based on drowsy sample ratio.
    
    If >=10% of samples are labeled as drowsy (1), label window as 1.
    This is more realistic for 16s windows where drowsy events are 2-4s.
    
    Parameters
    ----------
    labels_in_window : np.ndarray
        Labels for all samples in the window
    threshold : float
        Minimum ratio of drowsy samples (default: 0.1 = 10%)
        
    Returns
    -------
    label : int
        0 (awake) or 1 (drowsy)
    """
    drowsy_ratio = np.sum(labels_in_window == 1) / len(labels_in_window)
    return 1 if drowsy_ratio >= threshold else 0


def extract_features_from_file(filename, file_path):
    """
    Extract features from a single MAT file (preprocessed for Phase 1 CNN).
    
    Parameters
    ----------
    filename : str
        Name of the MAT file (e.g., "01M_1.mat")
    file_path : str
        Full path to the MAT file
        
    Returns
    -------
    records : list of dict
        List of feature records (one per window)
    """
    # Load data from preprocessed MAT file
    E1, E2, labels, fs = load_mat_file(file_path)
    if E1 is None:
        return []
    
    # Parse subject and trial from filename
    # Format: XXX_Y.mat (e.g., 01M_1.mat, 07F_2.mat)
    subject = filename[:3]  # e.g., "01M"
    trial = int(filename[4])  # e.g., 1 or 2
    
    # Create sliding windows
    windows = create_sliding_windows(len(E1), WINDOW_SAMPLES, STRIDE)
    
    records = []
    for window_idx, (start, end) in enumerate(windows):
        # Extract window data
        loc_window = E1[start:end]
        roc_window = E2[start:end]
        labels_window = labels[start:end]
        
        # Skip incomplete windows
        if len(loc_window) < WINDOW_SAMPLES:
            continue
        
        # Extract features
        try:
            features = extract_window_features(loc_window, roc_window, FS)
            
            # Assign label
            label = assign_window_label(labels_window)
            
            # Create record
            record = {
                'Subject': subject,
                'Trial': trial,
                'Window': window_idx,
                'StartSample': start,
                'EndSample': end,
                'Label': label
            }
            record.update(features)
            
            records.append(record)
            
        except Exception as e:
            print(f"  Warning: Failed to extract features for window {window_idx} in {filename}: {e}")
            continue
    
    return records


def process_file_set(files, set_name):
    """
    Process a set of files and extract features.
    
    Parameters
    ----------
    files : list
        List of .mat filenames
    set_name : str
        Name of the set (e.g., 'train', 'test')
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing all extracted features
    """
    print(f"\n{'='*60}")
    print(f"Processing {set_name} set ({len(files)} files)")
    print(f"{'='*60}")
    
    all_records = []
    
    for filename in tqdm(files, desc=f"Extracting {set_name} features"):
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"  Warning: File not found: {file_path}")
            continue
        
        records = extract_features_from_file(filename, file_path)
        all_records.extend(records)
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Print summary
    print(f"\n{set_name.upper()} SET SUMMARY:")
    print(f"  Total windows: {len(df)}")
    
    if len(df) > 0:
        print(f"  Drowsy windows: {np.sum(df['Label'] == 1)} ({100*np.mean(df['Label']==1):.2f}%)")
        print(f"  Awake windows: {np.sum(df['Label'] == 0)} ({100*np.mean(df['Label']==0):.2f}%)")
        print(f"  Features extracted: {len(df.columns) - 6}")  # Minus metadata columns
    else:
        print(f"  WARNING: No data extracted!")
    
    return df


def main():
    """Main feature extraction pipeline"""
    print("="*60)
    print("EOG Feature Extraction Pipeline")
    print("="*60)
    print(f"Window size: {WINDOW_SIZE}s ({WINDOW_SAMPLES} samples)")
    print(f"Stride: {STRIDE} samples ({STRIDE/FS:.2f}s)")
    print(f"Sampling frequency: {FS} Hz")
    print("="*60)
    
    # Load file sets
    files_train, files_test = load_file_sets()
    
    # Extract features for train set
    df_train = process_file_set(files_train, 'train')
    train_output = os.path.join(OUTPUT_DIR, 'train_features_16s.csv')
    df_train.to_csv(train_output, index=False)
    print(f"\n[OK] Train features saved to: {train_output}")
    print(f"  Size: {os.path.getsize(train_output) / (1024**2):.2f} MB")
    
    # Extract features for test set
    df_test = process_file_set(files_test, 'test')
    test_output = os.path.join(OUTPUT_DIR, 'test_features_16s.csv')
    df_test.to_csv(test_output, index=False)
    print(f"\n[OK] Test features saved to: {test_output}")
    print(f"  Size: {os.path.getsize(test_output) / (1024**2):.2f} MB")
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review feature CSVs in: {OUTPUT_DIR}")
    print(f"2. Run train_ml_colab.ipynb in Google Colab")
    print(f"3. Compare results with CNN_16s on Subject 07")
    print("="*60)


if __name__ == "__main__":
    main()

