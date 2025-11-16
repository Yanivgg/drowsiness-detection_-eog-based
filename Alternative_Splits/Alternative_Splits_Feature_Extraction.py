"""
Alternative Splits Feature Extraction for ML Training
=====================================================

This script extracts features for Alternative Splitting strategies:
1. Split 1.5 Recordings: Train on _1 (full) + _2 (first 50%), Test on _2 (last 50%)
2. Within-Subject Split: Train on _1 (full), Test on _2 (full)

Uses the same feature engineering as Phase2_ML for fair comparison.

Configuration:
- Window: 16 seconds (3200 samples @ 200 Hz) - matches CNN_16s
- Stride: 8 seconds (1600 samples, 50% overlap) - matches Phase2
- Features: ~63 features (time, frequency, non-linear, EOG-specific)

Author: Yaniv Grosberg
Date: 2025-11-16
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as spio
from tqdm import tqdm
from datetime import datetime

# Add Phase2_ML to path for feature engineering imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE2_DIR = os.path.join(BASE_DIR, 'Phase2_ML')
sys.path.insert(0, PHASE2_DIR)

from feature_engineering.time_domain import extract_time_domain_features
from feature_engineering.frequency_domain import extract_frequency_domain_features
from feature_engineering.nonlinear import extract_nonlinear_features
from feature_engineering.eog_specific import extract_eog_specific_features

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
DATA_DIR = os.path.join(BASE_DIR, 'Phase1_CNN', 'data', 'processed', 'files')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_features')

# Window parameters (same as Phase2_ML)
WINDOW_SIZE = 16  # seconds
WINDOW_SAMPLES = 3200  # samples (16s * 200 Hz)
STRIDE = 1600  # samples (8 second stride = 50% overlap)
FS = 200  # sampling frequency

# Subjects
ALL_SUBJECTS = ['01M', '02F', '03F', '04M', '05M', '06M', '07F', '08M', '09M', '10M']

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("Alternative Splits Feature Extraction for ML Training")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nConfiguration:")
print(f"  Window: {WINDOW_SIZE}s ({WINDOW_SAMPLES} samples)")
print(f"  Stride: {STRIDE} samples ({STRIDE/FS:.1f}s, {100*STRIDE/WINDOW_SAMPLES:.0f}% overlap)")
print(f"  Sampling frequency: {FS} Hz")
print(f"  Data directory: {DATA_DIR}")
print(f"  Output directory: {OUTPUT_DIR}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_mat_file(filepath, split_type='full'):
    """
    Load a single .mat file with optional split for _2 recordings.
    
    Parameters
    ----------
    filepath : str
        Full path to the MAT file
    split_type : str
        'full', 'first_half', or 'second_half'
        Only applies to _2 recordings
        
    Returns
    -------
    sig_loc : np.ndarray
        LOC channel signal
    sig_roc : np.ndarray
        ROC channel signal
    labels : np.ndarray
        Binary labels (0=awake, 1=drowsy)
    fs : int
        Sampling frequency
    """
    try:
        # Load MAT file
        mat_data = spio.loadmat(filepath)
        
        if 'Data' not in mat_data:
            print(f"  Warning: 'Data' field not found in {filepath}")
            return None, None, None, None
        
        data = mat_data['Data']
        
        # Extract EOG channels and labels
        sig_loc = data['E1'][0, 0].squeeze()  # LOC channel
        sig_roc = data['E2'][0, 0].squeeze()  # ROC channel
        labels = data['labels_O1'][0, 0].squeeze()  # Labels
        
        # Verify data
        if len(sig_loc) != len(sig_roc) or len(sig_loc) != len(labels):
            print(f"  Warning: Signal and label lengths don't match in {filepath}")
            return None, None, None, None
        
        # Apply split if needed (only for _2 files)
        filename = os.path.basename(filepath)
        if split_type != 'full' and filename.endswith('_2.mat'):
            split_point = len(sig_loc) // 2
            
            if split_type == 'first_half':
                sig_loc = sig_loc[:split_point]
                sig_roc = sig_roc[:split_point]
                labels = labels[:split_point]
            elif split_type == 'second_half':
                sig_loc = sig_loc[split_point:]
                sig_roc = sig_roc[split_point:]
                labels = labels[split_point:]
        
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


def extract_features_from_file(filename, file_path, split_type='full'):
    """
    Extract features from a single MAT file with optional split.
    
    Parameters
    ----------
    filename : str
        Name of the MAT file (e.g., "01M_1.mat")
    file_path : str
        Full path to the MAT file
    split_type : str
        'full', 'first_half', or 'second_half'
        
    Returns
    -------
    records : list of dict
        List of feature records (one per window)
    """
    # Load data from preprocessed MAT file
    E1, E2, labels, fs = load_mat_file(file_path, split_type)
    if E1 is None:
        return []
    
    # Parse subject and trial from filename
    # Format: XXX_Y.mat (e.g., 01M_1.mat, 07F_2.mat)
    subject = filename[:3]  # e.g., "01M"
    trial = int(filename[4])  # e.g., 1 or 2
    
    # Create sliding windows
    windows = create_sliding_windows(len(E1), WINDOW_SAMPLES, STRIDE)
    
    if len(windows) == 0:
        print(f"  Warning: No windows created for {filename} (split_type={split_type})")
        return []
    
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


def process_file_set(file_configs, set_name):
    """
    Process a set of files and extract features.
    
    Parameters
    ----------
    file_configs : list of tuples
        Each tuple is (filename, split_type)
    set_name : str
        Name of the set (for logging)
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with all extracted features
    """
    print(f"\n{'='*80}")
    print(f"Processing {set_name} set")
    print(f"{'='*80}")
    print(f"Files: {len(file_configs)}")
    
    all_records = []
    
    for filename, split_type in tqdm(file_configs, desc=f"Extracting {set_name}"):
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"  Warning: File not found: {file_path}")
            continue
        
        records = extract_features_from_file(filename, file_path, split_type)
        all_records.extend(records)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"{set_name} Set Statistics")
    print(f"{'='*80}")
    print(f"Total windows: {len(df):,}")
    if len(df) > 0:
        print(f"Awake windows: {(df['Label'] == 0).sum():,} ({100*(df['Label'] == 0).sum()/len(df):.2f}%)")
        print(f"Drowsy windows: {(df['Label'] == 1).sum():,} ({100*(df['Label'] == 1).sum()/len(df):.2f}%)")
        print(f"Features extracted: {len(df.columns) - 6}")  # Exclude metadata columns
    
    return df


# ============================================================
# EXPERIMENT 1: Split 1.5 Recordings
# ============================================================

def extract_split_1_5():
    """
    Extract features for Split 1.5 Recordings strategy.
    
    Train: All _1 (full) + All _2 (first 50%)
    Test: All _2 (last 50%)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Split 1.5 Recordings")
    print("="*80)
    print("Train: All _1 recordings (full) + All _2 recordings (first 50%)")
    print("Test: All _2 recordings (last 50%)")
    
    # Train files
    train_configs = []
    # Add all _1 files (full)
    for subject in ALL_SUBJECTS:
        train_configs.append((f"{subject}_1.mat", 'full'))
    # Add all _2 files (first half only)
    for subject in ALL_SUBJECTS:
        train_configs.append((f"{subject}_2.mat", 'first_half'))
    
    # Test files
    test_configs = []
    # Add all _2 files (second half only)
    for subject in ALL_SUBJECTS:
        test_configs.append((f"{subject}_2.mat", 'second_half'))
    
    # Extract features
    train_df = process_file_set(train_configs, "Split 1.5 Train")
    test_df = process_file_set(test_configs, "Split 1.5 Test")
    
    # Save to CSV
    train_path = os.path.join(OUTPUT_DIR, 'split_1_5_train_features_16s.csv')
    test_path = os.path.join(OUTPUT_DIR, 'split_1_5_test_features_16s.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Saved train features to: {train_path}")
    print(f"✓ Saved test features to: {test_path}")
    
    return train_df, test_df


# ============================================================
# EXPERIMENT 2: Within-Subject Split
# ============================================================

def extract_within_subject():
    """
    Extract features for Within-Subject Split strategy.
    
    Train: All _1 (full)
    Test: All _2 (full)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Within-Subject Split")
    print("="*80)
    print("Train: All _1 recordings (full)")
    print("Test: All _2 recordings (full)")
    
    # Train files (all _1)
    train_configs = []
    for subject in ALL_SUBJECTS:
        train_configs.append((f"{subject}_1.mat", 'full'))
    
    # Test files (all _2)
    test_configs = []
    for subject in ALL_SUBJECTS:
        test_configs.append((f"{subject}_2.mat", 'full'))
    
    # Extract features
    train_df = process_file_set(train_configs, "Within-Subject Train")
    test_df = process_file_set(test_configs, "Within-Subject Test")
    
    # Save to CSV
    train_path = os.path.join(OUTPUT_DIR, 'within_subject_train_features_16s.csv')
    test_path = os.path.join(OUTPUT_DIR, 'within_subject_test_features_16s.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Saved train features to: {train_path}")
    print(f"✓ Saved test features to: {test_path}")
    
    return train_df, test_df


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Verifying data directory...")
    print("="*80)
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("\nPlease ensure Phase1_CNN processed data exists.")
        sys.exit(1)
    
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
    print(f"✓ Found {len(mat_files)} .mat files")
    
    if len(mat_files) != 20:
        print(f"⚠️  Expected 20 files, found {len(mat_files)}")
    
    # Extract features for both experiments
    print("\n" + "="*80)
    print("Starting Feature Extraction")
    print("="*80)
    
    # Experiment 1: Split 1.5
    split_1_5_train, split_1_5_test = extract_split_1_5()
    
    # Experiment 2: Within-Subject
    within_train, within_test = extract_within_subject()
    
    # Summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nGenerated Files:")
    print(f"  1. split_1_5_train_features_16s.csv ({len(split_1_5_train):,} windows)")
    print(f"  2. split_1_5_test_features_16s.csv ({len(split_1_5_test):,} windows)")
    print(f"  3. within_subject_train_features_16s.csv ({len(within_train):,} windows)")
    print(f"  4. within_subject_test_features_16s.csv ({len(within_test):,} windows)")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\n✓ Ready for ML training in Colab!")
    print("="*80)

