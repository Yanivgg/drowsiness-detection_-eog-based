# -*- coding: utf-8 -*-
"""
Feature Extraction for Dryad EOG Dataset (v2)
---------------------------------------------
Updated version supporting modular window creation and flexible labeling logic.

Uses helper functions from:
    window_label_utils.py  -> create_windows(), assign_labels_and_filter()

Author: Yaniv Grosberg
Date: 2025-11-05
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, welch
import pyedflib
from window_label_utils import create_windows, assign_labels_and_filter_v2 as assign_labels_and_filter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = r"C:\Yaniv2\dataset_dryad"

# Signal processing
BAND = (0.1, 10)   # Hz - EOG frequency band
ORDER = 4          # Filter order

# Windowing & labeling configuration
config = {
    "window_sec": 15,             # seconds per window
    "overlap": 0.0,               # window overlap (0.0 = none, 0.5 = 50%)
    "pre_label_extend": 0,        # label 2 windows before each annotation
    "post_label_extend": 0,       # label 1 window after each annotation
    "pre_remove": 3,              # remove 1 window before each annotation
    "post_remove": 2,             # remove 1 window after each annotation
    "remove_first_last_ann": True,# remove first and last annotation
    "label_mode": "center"        # "center", "before", or "after"
}

def build_output_filename(base_dir, base_name, config):
    """
    Build a descriptive output filename based on the experiment configuration.
    Example:
        dryad_features_win10_ov0.5_pre2_post1_mode_before.csv
    """
    parts = [
        f"win{config['window_sec']}",
        f"ov{int(config.get('overlap', 0)*100)}",
        f"preL{config.get('pre_label_extend', 0)}",
        f"postL{config.get('post_label_extend', 0)}",
        f"preR{config.get('pre_remove', 0)}",
        f"postR{config.get('post_remove', 0)}",
        f"mode_{config.get('label_mode', 'center')}"
    ]
    fname = f"{base_name}_{'_'.join(parts)}.csv"
    return os.path.join(base_dir, fname)

OUT_DIR = r"C:\Yaniv2\datadryad\output_files"
BASE_NAME = "dryad_features_v2"
OUT_PATH = build_output_filename(OUT_DIR, BASE_NAME, config)
OUT_DIR = r"C:\Yaniv2\datadryad\output_files"
BASE_NAME = "dryad_features_v2"
OUT_PATH = build_output_filename(OUT_DIR, BASE_NAME, config)


# ============================================================
# FILTER UTILITIES
# ============================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_shape_features(signal):
    """Extract shape-based features from a 1D EOG segment."""
    duration = len(signal)
    peak_amplitude = np.max(signal)
    valley_amplitude = np.min(signal)
    mean_amplitude = np.mean(signal)
    time_to_peak = np.argmax(signal)
    time_to_valley = np.argmin(signal)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    zero_crossing_rate = len(zero_crossings) / duration

    signal_slope = np.gradient(signal)
    rise_time = np.where(signal >= peak_amplitude)[0][0]
    fall_time = duration - np.where(signal[::-1] >= peak_amplitude)[0][0]
    signal_curvature = np.gradient(signal_slope)
    inflection_points = np.where(np.diff(np.sign(signal_curvature)))[0]

    crest_factor = peak_amplitude / np.sqrt(np.mean(np.square(signal)))
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)
    std_deviation = np.std(signal)
    variance = np.var(signal)
    median = np.median(signal)
    min_to_max_ratio = valley_amplitude / peak_amplitude if peak_amplitude != 0 else 0
    mean_abs_deviation = np.mean(np.abs(signal - mean_amplitude))
    area_under_curve = np.trapezoid(signal)

    return [
        peak_amplitude, valley_amplitude, mean_amplitude,
        time_to_peak, time_to_valley, rise_time, fall_time,
        len(inflection_points), crest_factor, zero_crossing_rate,
        skew_val, kurt_val, std_deviation, variance, median,
        min_to_max_ratio, mean_abs_deviation, area_under_curve
    ]


def bandpower(signal, fs, band):
    f, Pxx = welch(signal, fs=fs, nperseg=fs*2)
    freq_res = f[1] - f[0]
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx_band], dx=freq_res)


def extract_frequency_features(signal, fs):
    """Extract bandpower features in standard EEG/EOG frequency bands."""
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30)
    }
    features = {}
    total_power = bandpower(signal, fs, (0.5, 30))
    for name, (low, high) in bands.items():
        p = bandpower(signal, fs, (low, high))
        features[f"bp_{name}"] = p
        features[f"rel_{name}"] = p / total_power if total_power > 0 else 0
    return features


# ============================================================
# MAIN PIPELINE
# ============================================================

records = []
edf_files = [f for f in os.listdir(BASE_PATH) if f.lower().endswith(".edf") and "annotations" not in f.lower()]

for f in sorted(edf_files):
    subj, trial = re.findall(r"(\d{2}[MF])_(\d)", f)[0]
    file_path = os.path.join(BASE_PATH, f)
    annot_path = file_path.replace(".edf", "_annotations.edf")

    # --- Read EDF ---
    with pyedflib.EdfReader(file_path) as reader:
        ch_names = reader.getSignalLabels()
        fs = reader.getSampleFrequency(0)
        dur = reader.file_duration
        n_samples = int(dur * fs)

        # Identify EOG channels
        eog_indices = [i for i, n in enumerate(ch_names) if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
        if len(eog_indices) < 2:
            continue

        sig_loc = butter_bandpass_filter(reader.readSignal(eog_indices[0]), *BAND, fs, ORDER)
        sig_roc = butter_bandpass_filter(reader.readSignal(eog_indices[1]), *BAND, fs, ORDER)

    # Differential signal (horizontal EOG)
    diff = sig_loc - sig_roc

    # --- Create windows ---
    windows, n_windows = create_windows(signal_length=n_samples, fs=fs,
                                        window_sec=config["window_sec"], overlap=config["overlap"])

    # --- Load and process annotations ---
    annots = []
    if os.path.exists(annot_path):
        with pyedflib.EdfReader(annot_path) as ann_reader:
            onsets, durations, descs = ann_reader.readAnnotations()
            annots = list(zip(onsets, durations, descs))

    labels, to_remove = assign_labels_and_filter(n_windows, annots, config)

    # --- Feature extraction per window ---
    feature_names = [
        "peak_amp", "valley_amp", "mean_amp", "time_to_peak", "time_to_valley",
        "rise_time", "fall_time", "n_inflect", "crest_factor", "zero_cross_rate",
        "skew", "kurt", "std", "var", "median", "min_max_ratio",
        "mean_abs_dev", "auc"
    ]

    for w, (t_start, t_end, s_start, s_end) in enumerate(windows):
        if w in to_remove:
            continue  # skip removed windows

        seg_loc = sig_loc[s_start:s_end]
        seg_roc = sig_roc[s_start:s_end]
        seg_diff = diff[s_start:s_end]

        # Skip empty or invalid windows
        if len(seg_diff) < fs * config["window_sec"] * 0.9:
            continue

        shape_feats = extract_shape_features(seg_diff)
        freq_feats = extract_frequency_features(seg_diff, fs)
        corr = np.corrcoef(seg_loc, seg_roc)[0, 1]

        record = {
            "Subject": subj,
            "Trial": int(trial),
            "Window": w,
            "StartTime_s": t_start,
            "EndTime_s": t_end,
            "Label": int(labels[w]),
            "Correlation_LOC_ROC": corr,
        }

        record.update(dict(zip(feature_names, shape_feats)))
        record.update(freq_feats)
        records.append(record)

# --- Save results ---
df = pd.DataFrame(records)
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ Extracted {len(df)} windows from {len(edf_files)} EDF files")
print(f"Saved feature dataset to: {OUT_PATH}")
