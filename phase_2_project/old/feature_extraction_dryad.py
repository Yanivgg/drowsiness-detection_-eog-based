# -*- coding: utf-8 -*-
"""
Feature Extraction for Dryad EOG Dataset
Based on Omerbzkt EOG-Classification and adapted for EDF + annotations.
Author: Yaniv Grosberg
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, welch
import pyedflib

# ============ CONFIG ============
BASE_PATH = r"C:\Yaniv2\dataset_dryad"
OUT_PATH = os.path.join(BASE_PATH, "dryad_features.csv")
WINDOW_SEC = 10     # Window size in seconds
BAND = (0.1, 10)    # EOG frequency band
ORDER = 4

# ============ FILTER ============

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, data)

# ============ FEATURE EXTRACTION (from Omerbzkt) ============

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

# ============ POWER FEATURES ============
def bandpower(signal, fs, band):
    f, Pxx = welch(signal, fs=fs, nperseg=fs*2)
    freq_res = f[1] - f[0]
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx_band], dx=freq_res)

def extract_frequency_features(signal, fs):
    """Extract bandpower features in common EEG/EOG ranges."""
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

# ============ MAIN PIPELINE ============
records = []

edf_files = [f for f in os.listdir(BASE_PATH) if f.lower().endswith(".edf") and "annotations" not in f.lower()]

for f in sorted(edf_files):
    subj, trial = re.findall(r"(\d{2}[MF])_(\d)", f)[0]
    file_path = os.path.join(BASE_PATH, f)
    annot_path = file_path.replace(".edf", "_annotations.edf")

    with pyedflib.EdfReader(file_path) as reader:
        ch_names = reader.getSignalLabels()
        fs = reader.getSampleFrequency(0)
        dur = reader.file_duration
        n_samples = int(dur * fs)

        eog_indices = [i for i, n in enumerate(ch_names) if any(k in n.upper() for k in ["EOG","LOC","ROC"])]
        if len(eog_indices) < 2:
            continue
        sig_loc = butter_bandpass_filter(reader.readSignal(eog_indices[0]), *BAND, fs, ORDER)
        sig_roc = butter_bandpass_filter(reader.readSignal(eog_indices[1]), *BAND, fs, ORDER)

    # Differential signal
    diff = sig_loc - sig_roc

    # Divide into windows
    samples_per_win = int(WINDOW_SEC * fs)
    n_windows = n_samples // samples_per_win

    # Load annotations
    label = np.zeros(n_windows, dtype=int)
    if os.path.exists(annot_path):
        with pyedflib.EdfReader(annot_path) as ann_reader:
            onsets, durations, descs = ann_reader.readAnnotations()
            for onset, dur_ann, desc in zip(onsets, durations, descs):
                start_idx = int(onset // WINDOW_SEC)
                end_idx = int((onset + dur_ann) // WINDOW_SEC)
                for idx in range(start_idx, min(end_idx+1, n_windows)):
                    label[idx] = 1  # 1 = drowsy

    # Feature extraction per window
    for w in range(n_windows):
        start = w * samples_per_win
        end = start + samples_per_win
        seg_loc = sig_loc[start:end]
        seg_roc = sig_roc[start:end]
        seg_diff = diff[start:end]

        # Extract shape-based features (diff)
        shape_feats = extract_shape_features(seg_diff)

        # Extract frequency features
        freq_feats = extract_frequency_features(seg_diff, fs)

        # Correlation between LOC and ROC
        corr = np.corrcoef(seg_loc, seg_roc)[0,1]

        record = {
            "Subject": subj,
            "Trial": int(trial),
            "Window": w,
            "Label": int(label[w]),
            "Correlation_LOC_ROC": corr,
        }
        # Merge all features
        feature_names = [
            "peak_amp","valley_amp","mean_amp","time_to_peak","time_to_valley",
            "rise_time","fall_time","n_inflect","crest_factor","zero_cross_rate",
            "skew","kurt","std","var","median","min_max_ratio",
            "mean_abs_dev","auc"
        ]
        record.update(dict(zip(feature_names, shape_feats)))
        record.update(freq_feats)
        records.append(record)

# Create DataFrame and save
df = pd.DataFrame(records)
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ Extracted {len(df)} windows from {len(edf_files)} files")
print(f"Saved features to {OUT_PATH}")
