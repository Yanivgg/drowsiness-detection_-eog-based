# -*- coding: utf-8 -*-
"""
window_label_utils.py
=====================

Utility functions for creating, labeling, and filtering fixed-size windows
around annotations in EOG/EEG or other continuous recordings.

Features:
- Flexible window creation with optional overlap
- Labeling around annotations (before / after / centered)
- Excluding specific windows around annotations
- Option to skip the first and last annotation

Author: Yaniv Grosberg
Date: 2025-11-05
"""

import numpy as np


# ============================================================
# Window creation
# ============================================================

def create_windows(signal_length, fs, window_sec, overlap=0.0):
    """
    Divide a continuous signal into fixed-size windows with optional overlap.

    Parameters
    ----------
    signal_length : int
        Total number of samples in the signal.
    fs : int or float
        Sampling frequency (Hz).
    window_sec : float
        Window length in seconds.
    overlap : float, optional
        Overlap ratio between consecutive windows (0.0 = no overlap, 0.5 = 50% overlap).

    Returns
    -------
    windows : list of tuples
        Each tuple is (start_time_sec, end_time_sec, start_sample, end_sample).
    n_windows : int
        Total number of windows created.
    """
    win_size = int(window_sec * fs)
    step = int(win_size * (1 - overlap))
    windows = []

    for start in range(0, signal_length - win_size + 1, step):
        end = start + win_size
        start_time = start / fs
        end_time = end / fs
        windows.append((start_time, end_time, start, end))

    n_windows = len(windows)
    return windows, n_windows


# ============================================================
# Labeling and filtering
# ============================================================

def assign_labels_and_filter_v2(n_windows, annots, config):
    """
    Assign labels (0/1) to time windows based on annotations, while handling
    complex edge cases where labeled and removal windows may overlap.

    This version ensures two critical behaviors:
    1. Windows that are labeled (Label = 1) are **never removed**, even if they
       fall inside a removal range.
    2. Removal operations are applied **after** labeling, so no potential
       labeling opportunities are lost due to overlapping annotations.

    Parameters
    ----------
    n_windows : int
        Total number of windows in the current recording.
    annots : list of tuples
        List of annotations in the format:
        [(onset_sec, duration_sec, description), ...].
    config : dict
        Configuration dictionary with the following keys:
            - window_sec (float):
                Duration of each window in seconds.
            - pre_label_extend (int):
                Number of windows before each annotation to also label as 1.
            - post_label_extend (int):
                Number of windows after each annotation to also label as 1.
            - pre_remove (int):
                Number of windows before each annotation to remove entirely.
            - post_remove (int):
                Number of windows after each annotation to remove entirely.
            - remove_first_last_ann (bool):
                If True, skip the first and last annotation in each recording.
            - label_mode (str):
                Labeling strategy:
                    "center" – label the annotation window + extensions.
                    "before" – label only windows before the annotation.
                    "after"  – label only windows after the annotation.

    Returns
    -------
    labels : np.ndarray
        Binary array (0/1) with one label per window.
    to_remove : list of int
        Sorted list of window indices to exclude from the dataset.

    Notes
    -----
    - Labeled windows always take precedence over removal windows.
    - The removal operation is performed *after* labeling to avoid deleting
      windows that belong to neighboring annotations.
    - This logic prevents data loss when annotations are close together.
    """
    labels = np.zeros(n_windows, dtype=int)
    to_remove = set()
    labeled_windows = set()

    # Optionally skip first and last annotation
    if config.get("remove_first_last_ann", False) and len(annots) > 2:
        annots = annots[1:-1]

    # ---------------------------
    # PHASE 1: Labeling
    # ---------------------------
    for onset, dur, _ in annots:
        start_idx = int(onset // config["window_sec"])
        end_idx = int((onset + dur) // config["window_sec"])
        mode = config.get("label_mode", "center")

        if mode == "before":
            start_label = max(0, start_idx - config.get("pre_label_extend", 0))
            end_label = max(0, start_idx - 1)
        elif mode == "after":
            start_label = min(n_windows - 1, end_idx + 1)
            end_label = min(n_windows - 1, end_idx + config.get("post_label_extend", 0))
        else:  # "center" (default)
            start_label = max(0, start_idx - config.get("pre_label_extend", 0))
            end_label = min(n_windows - 1, end_idx + config.get("post_label_extend", 0))

        for i in range(start_label, end_label + 1):
            if 0 <= i < n_windows:
                labels[i] = 1
                labeled_windows.add(i)

    # ---------------------------
    # PHASE 2: Removal
    # ---------------------------
    for onset, dur, _ in annots:
        start_idx = int(onset // config["window_sec"])
        end_idx = int((onset + dur) // config["window_sec"])
        start_remove = max(0, start_idx - config.get("pre_remove", 0))
        end_remove = min(n_windows - 1, end_idx + config.get("post_remove", 0))
        for i in range(start_remove, end_remove + 1):
            # Skip removal if this window was labeled in phase 1
            if i not in labeled_windows:
                to_remove.add(i)

    return labels, sorted(list(to_remove))




# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example signal: 10 minutes sampled at 100 Hz
    fs = 100
    duration_sec = 600
    signal_length = fs * duration_sec

    # Create windows with 10 s length and 50% overlap
    windows, n_windows = create_windows(signal_length, fs, window_sec=10, overlap=0.5)
    print(f"Created {n_windows} windows.")

    # Example annotations
    annots_example = [
        (100, 10, "Drowsy"),
        (300, 20, "Drowsy"),
        (600, 15, "Alert")
    ]

    # Configuration
    cfg = {
        "window_sec": 10,
        "pre_label_extend": 2,
        "post_label_extend": 1,
        "pre_remove": 1,
        "post_remove": 1,
        "remove_first_last_ann": True,
        "label_mode": "before"  # "center", "before", "after"
    }

    # ✅ FIX: use the correct function name
    labels, remove_idx = assign_labels_and_filter_v2(n_windows, annots_example, cfg)

    print("Number of labeled windows (1):", np.sum(labels))
    print("Windows to remove:", remove_idx[:15], "...")
