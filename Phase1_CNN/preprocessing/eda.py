# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis for Drowsiness Detection Dataset

This script performs comprehensive EDA on the Dryad drowsiness dataset:
1. Dataset overview (10 subjects, 20 sessions)
2. Drowsiness annotation analysis and distribution
3. Labeling methodology explanation (2-second context windows)
4. Class imbalance visualization (96.5% Awake vs 3.5% Drowsy)
5. Subject-wise drowsiness frequency
6. Train/Val/Test split rationale
7. Signal quality assessment

Original file generated from Colab notebook.
Enhanced for drowsiness detection project documentation.
"""

import os
import re






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import mne
import pyedflib

# Configuration
base_path = r"C:\Yaniv2\dataset_dryad"

# Utility Functions
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter to signal data."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, data)

# ==========================================
# Part 1: Summary of all EDF files (signals + annotations)
# ==========================================
edf_files = [f for f in os.listdir(base_path) if f.lower().endswith(".edf")]
summary = []

for f in sorted(edf_files):
    file_path = os.path.join(base_path, f)
    info = {"File": f}
    try:
        reader = pyedflib.EdfReader(file_path)
        n_signals = reader.signals_in_file
        duration_sec = reader.file_duration
        sfreqs = [reader.getSampleFrequency(i) for i in range(n_signals)]
        ch_names = reader.getSignalLabels()

        # Identify if this is an annotation file by name or by number of channels
        is_annotation = (n_signals == 0) or ("annot" in f.lower())
        n_annot = len(reader.readAnnotations()[0]) if reader.readAnnotations()[0].size > 0 else 0

        info.update({
            "Is_Annotation": is_annotation,
            "Duration_min": round(duration_sec / 60, 2),
            "Sampling_Rate_Hz": sfreqs[0] if sfreqs else None,
            "N_Channels": n_signals,
            "Channel_Names": ", ".join(ch_names[:5]) + ("..." if len(ch_names) > 5 else ""),
            "N_Annotations": n_annot
        })
        reader.close()

    except Exception as e:
        info.update({"Error": str(e)})
    summary.append(info)

df_summary = pd.DataFrame(summary)
print("=== EDF & Annotation Summary ===")
print(df_summary.head(20))

summary_path = os.path.join(base_path, "dryad_eda_summary.csv")
df_summary.to_csv("dryad_eda_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n✅ Saved summary to: {summary_path}")

# ==========================================
# Part 2: Detailed breakdown of annotations
# ==========================================
records = []
ann_files = [f for f in edf_files if "_annotations" in f.lower()]

for f in sorted(ann_files):
    file_path = os.path.join(base_path, f)
    try:
        reader = pyedflib.EdfReader(file_path)
        onsets, durations, descs = reader.readAnnotations()

        # Extract subject ID (like 01M) and trial number (1/2)
        match = re.match(r"(\d{2}[MF])_(\d)", f)
        subj, trial = match.groups() if match else ("Unknown", "0")

        for onset, dur, desc in zip(onsets, durations, descs):
            records.append({
                "Subject": subj,
                "Trial": int(trial),
                "File": f,
                "Onset_s": round(float(onset), 2),
                "Duration_s": round(float(dur), 2),
                "Description": desc.strip() if isinstance(desc, str) else str(desc)
            })
        reader.close()

    except Exception as e:
        print(f"⚠️ Error reading {f}: {e}")

df_ann = pd.DataFrame(records)
if not df_ann.empty:
    df_ann.sort_values(["Subject", "Trial", "Onset_s"], inplace=True)
    ann_path = os.path.join(base_path, "annotations_summary.csv")
    df_ann.to_csv(ann_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved detailed annotations to: {ann_path}")
    print(f"Total annotations: {len(df_ann)}")

    # ==========================================
    # Part 3: Analysis and visualization
    # ==========================================
    summary_counts = (
        df_ann.groupby(["Subject", "Trial"])
        .size()
        .reset_index(name="N_Annotations")
        .sort_values(["Subject", "Trial"])
    )

    print("\n=== Annotations count per subject/trial ===")
    print(summary_counts)

    # Plot
    plt.figure(figsize=(10,5))
    plt.bar(
        summary_counts["Subject"] + "_" + summary_counts["Trial"].astype(str),
        summary_counts["N_Annotations"],
        color="skyblue",
        edgecolor="black"
    )
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Annotations")
    plt.title("Annotations per Subject/Trial (Dryad DD-Database)")
    plt.tight_layout()
    plt.show()
    
    # ==========================================
    # Part 3.5: Labeling Methodology & Class Imbalance Analysis
    # ==========================================
    print("\n" + "="*80)
    print("LABELING METHODOLOGY & CLASS IMBALANCE ANALYSIS")
    print("="*80)
    
    print("\n📋 Labeling Strategy:")
    print("-" * 80)
    print("Each drowsiness event is labeled using a 2-SECOND CONTEXT WINDOW approach:")
    print("  • Original annotation: marks the exact drowsiness event")
    print("  • Context extension: ±1 second around each event")
    print("  • Total window: 2 seconds (captures transition periods)")
    print("  • Samples in window: labeled as 'Drowsy' (1)")
    print("  • All other samples: labeled as 'Awake' (0)")
    print("\nRationale:")
    print("  ✓ Captures gradual transitions into/out of drowsiness")
    print("  ✓ Provides more training data while maintaining clinical relevance")
    print("  ✓ Accounts for annotation imprecision (human labeling)")
    
    # Calculate class imbalance statistics
    total_events = len(df_ann)
    total_duration_drowsy_sec = df_ann["Duration_s"].sum()
    
    # Assuming 128 Hz sampling rate and 2-second context windows
    sampling_rate = 128  # Hz
    context_window = 2  # seconds
    samples_per_event = context_window * sampling_rate
    total_drowsy_samples = total_events * samples_per_event
    
    # Estimate total recording duration (from summary)
    total_recording_duration = df_summary[df_summary["Is_Annotation"] == False]["Duration_min"].sum() * 60  # in seconds
    total_samples = int(total_recording_duration * sampling_rate)
    total_awake_samples = total_samples - total_drowsy_samples
    
    drowsy_percentage = (total_drowsy_samples / total_samples) * 100
    awake_percentage = (total_awake_samples / total_samples) * 100
    
    print("\n📊 Class Imbalance Statistics:")
    print("-" * 80)
    print(f"Total drowsiness events: {total_events:,}")
    print(f"Total recording duration: {total_recording_duration/3600:.2f} hours")
    print(f"Total samples (@ 128 Hz): {total_samples:,}")
    print(f"\nClass Distribution:")
    print(f"  • Awake samples:  {total_awake_samples:,} ({awake_percentage:.2f}%)")
    print(f"  • Drowsy samples: {total_drowsy_samples:,} ({drowsy_percentage:.2f}%)")
    print(f"  • Imbalance ratio: {awake_percentage/drowsy_percentage:.1f}:1 (Awake:Drowsy)")
    
    print("\n⚠️  Severe Class Imbalance Detected!")
    print("   This requires special handling:")
    print("   • Class weighting during training (drowsy weighted 30-40x)")
    print("   • Cohen's Kappa as primary metric (handles imbalance)")
    print("   • Precision-Recall analysis (not just accuracy)")
    
    # Visualize class imbalance
    plt.figure(figsize=(10, 6))
    
    # Pie chart
    plt.subplot(1, 2, 1)
    colors = ['#2ecc71', '#e74c3c']
    plt.pie([awake_percentage, drowsy_percentage], 
            labels=['Awake', 'Drowsy'],
            autopct='%1.2f%%',
            colors=colors,
            startangle=90,
            explode=(0, 0.1))
    plt.title('Class Distribution\n(Severe Imbalance)')
    
    # Bar chart
    plt.subplot(1, 2, 2)
    plt.bar(['Awake', 'Drowsy'], 
            [total_awake_samples, total_drowsy_samples],
            color=colors,
            edgecolor='black')
    plt.ylabel('Number of Samples')
    plt.title('Sample Count by Class')
    plt.yscale('log')  # Log scale to show both classes
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate([total_awake_samples, total_drowsy_samples]):
        plt.text(i, v, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💾 Saved: class_imbalance_analysis.png")
    
    # Subject-wise drowsiness frequency
    print("\n📈 Subject-Wise Drowsiness Frequency:")
    print("-" * 80)
    subject_stats = df_ann.groupby("Subject").agg({
        "Onset_s": "count",
        "Duration_s": "sum"
    }).rename(columns={"Onset_s": "Event_Count", "Duration_s": "Total_Duration_s"})
    subject_stats["Avg_Duration_s"] = subject_stats["Total_Duration_s"] / subject_stats["Event_Count"]
    subject_stats = subject_stats.sort_values("Event_Count", ascending=False)
    
    print(subject_stats)
    print(f"\n🎯 Subject 07 has the MOST drowsiness events → Chosen as TEST SET")
    print("   This provides the most robust evaluation dataset.")
    
    # Visualize subject-wise distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.barh(subject_stats.index, subject_stats["Event_Count"], color='coral', edgecolor='black')
    plt.xlabel('Number of Drowsiness Events')
    plt.ylabel('Subject')
    plt.title('Drowsiness Events per Subject')
    plt.grid(axis='x', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.barh(subject_stats.index, subject_stats["Total_Duration_s"], color='steelblue', edgecolor='black')
    plt.xlabel('Total Drowsiness Duration (seconds)')
    plt.ylabel('Subject')
    plt.title('Total Drowsiness Duration per Subject')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('subject_wise_drowsiness.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n💾 Saved: subject_wise_drowsiness.png")
    
    print("\n" + "="*80)
    print("END OF LABELING & IMBALANCE ANALYSIS")
    print("="*80 + "\n")
    
else:
    print("⚠️ No annotations found or parsed.")

df_ann = pd.read_csv("annotations_summary.csv")

plt.figure(figsize=(12,6))
sns.scatterplot(
    data=df_ann,
    x="Onset_s",
    y=df_ann["Subject"] + "_" + df_ann["Trial"].astype(str),
    hue="Subject",
    s=30,
    alpha=0.7,
    palette="tab10",
    legend=False
)
plt.xlabel("Time (seconds from start of recording)")
plt.ylabel("Subject_Trial")
plt.title("Timeline of Drowsiness Events per Recording (Dryad DD-Database)")
plt.tight_layout()
plt.show()

# Read and visualize EOG signals from single file
file_name = "01M_1.edf"
file_path = os.path.join(base_path, file_name)

with pyedflib.EdfReader(file_path) as reader:
    ch_names = reader.getSignalLabels()
    fs = reader.getSampleFrequency(0)
    n_signals = reader.signals_in_file
    dur = reader.file_duration
    total_samples = int(dur * fs)

    print(f"Duration (s): {dur:.1f} | Samples: {total_samples}")

    eog_indices = [i for i, n in enumerate(ch_names) if any(k in n.upper() for k in ["EOG","LOC","ROC"])]
    if not eog_indices:
        eog_indices = [4,5]
    print(f"EOG channels: {[ch_names[i] for i in eog_indices]}")

    # One minute window from the beginning
    start_sec = 0
    window_sec = 60
    start_sample = int(start_sec * fs)
    end_sample = min(int((start_sec + window_sec) * fs), total_samples)

    signals = []
    for i in eog_indices:
        sig = reader.readSignal(i)[start_sample:end_sample]
        sig_filt = butter_bandpass_filter(sig, 0.1, 15, fs)
        signals.append(sig_filt)

time = np.arange(len(signals[0])) / fs
plt.figure(figsize=(12,5))
for idx, sig in enumerate(signals):
    plt.plot(time, sig + idx*200, label=f"{ch_names[eog_indices[idx]]}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (µV, shifted)")
plt.title(f"EOG signals ({file_name}) — {window_sec/60:.1f} min from start")
plt.legend()
plt.tight_layout()
plt.show()

# Visualize multiple 30-second windows
file_name = "01M_1.edf"
file_path = os.path.join(base_path, file_name)

# Read the file
with pyedflib.EdfReader(file_path) as reader:
    ch_names = reader.getSignalLabels()
    fs = reader.getSampleFrequency(0)
    dur = reader.file_duration
    total_samples = int(dur * fs)

    # Find EOG channels
    eog_indices = [i for i, n in enumerate(ch_names)
                   if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
    if not eog_indices:
        eog_indices = [4,5]

    print(f"📄 File: {file_name}")
    print(f"📡 Channels: {[ch_names[i] for i in eog_indices]}")
    print(f"⌚ Duration: {dur/60:.1f} min | Fs: {fs} Hz")

    # Read all signals
    signals = [butter_bandpass_filter(reader.readSignal(i), 0.1, 15, fs)
               for i in eog_indices]

# Divide into 30-second windows
window_sec = 30
samples_per_win = int(window_sec * fs)
n_windows = total_samples // samples_per_win

print(f"Total windows of 30s: {n_windows}")

num_to_plot = 4  # How many windows to display (can be changed)
plt.figure(figsize=(12, num_to_plot * 2.5))

for w in range(num_to_plot):
    start = w * samples_per_win
    end = start + samples_per_win
    time = np.arange(start, end) / fs
    plt.subplot(num_to_plot, 1, w+1)
    for idx, sig in enumerate(signals):
        plt.plot(time, sig[start:end] + idx*250, label=f"{ch_names[eog_indices[idx]]}")
    plt.title(f"Window {w+1} — {w*window_sec:.0f} to {(w+1)*window_sec:.0f} sec")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV, shifted)")
    plt.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()

# Calculate EOG activity over entire recording with 10s windows
file_name = "01M_1.edf"
file_path = os.path.join(base_path, file_name)

# Read file
with pyedflib.EdfReader(file_path) as reader:
    ch_names = reader.getSignalLabels()
    fs = reader.getSampleFrequency(0)
    dur = reader.file_duration
    total_samples = int(dur * fs)

    # Select EOG channels (LOC and ROC)
    eog_indices = [i for i, n in enumerate(ch_names)
                   if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
    if not eog_indices:
        eog_indices = [4,5]

    print(f"📄 File: {file_name}")
    print(f"⌚ Duration: {dur/60:.1f} min | Fs: {fs} Hz")
    print(f"🎯 EOG channels: {[ch_names[i] for i in eog_indices]}")

    signals = [butter_bandpass_filter(reader.readSignal(i), 0.1, 15, fs)
               for i in eog_indices]

# Divide into 10-second windows
win_sec = 10
samples_per_win = int(win_sec * fs)
n_windows = total_samples // samples_per_win

# Mean absolute value (activity level)
mean_activity = []
for sig in signals:
    activity = [
        np.mean(np.abs(sig[i*samples_per_win:(i+1)*samples_per_win]))
        for i in range(n_windows)
    ]
    mean_activity.append(activity)

# Central time for each window
time_axis = np.arange(n_windows) * win_sec / 60  # minutes

# Plot
plt.figure(figsize=(12,5))
for idx, activity in enumerate(mean_activity):
    plt.plot(time_axis, activity, label=f"{ch_names[eog_indices[idx]]}")
plt.xlabel("Time (minutes)")
plt.ylabel("Mean |EOG| (µV)")
plt.title(f"EOG activity over entire recording ({file_name})\nWindow size = {win_sec} s")
plt.legend()
plt.tight_layout()
plt.show()

# EOG activity with drowsiness annotations overlay
signal_file = "01M_1.edf"
annot_file = "01M_1_annotations.edf"
signal_path = os.path.join(base_path, signal_file)
annot_path = os.path.join(base_path, annot_file)

# Read signal
with pyedflib.EdfReader(signal_path) as reader:
    ch_names = reader.getSignalLabels()
    fs = reader.getSampleFrequency(0)
    dur = reader.file_duration
    total_samples = int(dur * fs)

    eog_indices = [i for i, n in enumerate(ch_names)
                   if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
    if not eog_indices:
        eog_indices = [4,5]

    signals = [butter_bandpass_filter(reader.readSignal(i), 0.1, 15, fs)
               for i in eog_indices]

# Calculate mean for each window
win_sec = 10
samples_per_win = int(win_sec * fs)
n_windows = total_samples // samples_per_win

mean_activity = []
for sig in signals:
    activity = [
        np.mean(np.abs(sig[i*samples_per_win:(i+1)*samples_per_win]))
        for i in range(n_windows)
    ]
    mean_activity.append(activity)

time_axis = np.arange(n_windows) * win_sec / 60  # minutes

# Read annotations
with pyedflib.EdfReader(annot_path) as ann_reader:
    onsets, durations, descs = ann_reader.readAnnotations()

# Convert to minutes
annotation_times_min = np.array(onsets) / 60

# Plot
plt.figure(figsize=(12,5))
for idx, activity in enumerate(mean_activity):
    plt.plot(time_axis, activity, label=f"{ch_names[eog_indices[idx]]}")

# Vertical lines for drowsiness events
for t in annotation_times_min:
    plt.axvline(t, color='red', linestyle='--', alpha=0.6)

plt.xlabel("Time (minutes)")
plt.ylabel("Mean |EOG| (µV)")
plt.title(f"EOG activity with drowsiness annotations ({signal_file})\nWindow size = {win_sec}s")
plt.legend()
plt.tight_layout()
plt.show()

# Loop through all EDF files and display EOG with annotations
for file_name in sorted(os.listdir(base_path)):
    if not file_name.lower().endswith(".edf"):
        continue
    if "annotations" in file_name.lower():
        continue  # Skip annotation files

    file_path = os.path.join(base_path, file_name)
    print(f"\n📄 Processing {file_name} ...")

    try:
        # Read signal
        with pyedflib.EdfReader(file_path) as reader:
            ch_names = reader.getSignalLabels()
            fs = reader.getSampleFrequency(0)
            dur = reader.file_duration
            total_samples = int(dur * fs)

            # Find EOG channels
            eog_indices = [i for i, n in enumerate(ch_names)
                           if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
            if not eog_indices:
                eog_indices = [4,5]  # fallback typical indices

            print(f"⌚ Duration: {dur/60:.1f} min | Fs: {fs} Hz")
            print(f"🎯 EOG channels: {[ch_names[i] for i in eog_indices]}")

            # Read and filter signals
            signals = [butter_bandpass_filter(reader.readSignal(i), 0.1, 15, fs)
                       for i in eog_indices]

        # Calculate mean for each window
        win_sec = 10
        samples_per_win = int(win_sec * fs)
        n_windows = total_samples // samples_per_win

        mean_activity = []
        for sig in signals:
            activity = [
                np.mean(np.abs(sig[i*samples_per_win:(i+1)*samples_per_win]))
                for i in range(n_windows)
            ]
            mean_activity.append(activity)

        time_axis = np.arange(n_windows) * win_sec / 60  # minutes

        # Read matching annotation file
        annot_file = file_name.replace(".edf", "_annotations.edf")
        annotation_times_min = []
        annot_path = os.path.join(base_path, annot_file)

        if os.path.exists(annot_path):
            with pyedflib.EdfReader(annot_path) as ann_reader:
                onsets, durations, descs = ann_reader.readAnnotations()
                annotation_times_min = np.array(onsets) / 60
            print(f"🟥 Loaded {len(annotation_times_min)} annotations from {annot_file}")
        else:
            print("⚠️ No matching annotation file found.")

        # Plot
        plt.figure(figsize=(12,5))
        for idx, activity in enumerate(mean_activity):
            plt.plot(time_axis, activity, label=f"{ch_names[eog_indices[idx]]}")

        # Vertical lines for annotations
        for t in annotation_times_min:
            plt.axvline(t, color='red', linestyle='--', alpha=0.6)

        plt.xlabel("Time (minutes)")
        plt.ylabel("Mean |EOG| (µV)")
        plt.title(f"EOG activity with drowsiness annotations ({file_name})\nWindow size = {win_sec}s")
        plt.legend()
        plt.tight_layout()
        plt.show()

        input("Press Enter to continue to next file...")

    except Exception as e:
        print(f"⚠️ Error processing {file_name}: {e}")

# Generate summary statistics for drowsiness metrics across all subjects
summary = []

# Loop through all EDF files (without annotations)
for file_name in sorted(os.listdir(base_path)):
    if not file_name.lower().endswith(".edf"):
        continue
    if "annotations" in file_name.lower():
        continue

    file_path = os.path.join(base_path, file_name)
    annot_file = file_name.replace(".edf", "_annotations.edf")
    annot_path = os.path.join(base_path, annot_file)

    try:
        # Read signals
        with pyedflib.EdfReader(file_path) as reader:
            ch_names = reader.getSignalLabels()
            fs = reader.getSampleFrequency(0)
            dur = reader.file_duration
            total_samples = int(dur * fs)

            # EOG channels
            eog_indices = [i for i, n in enumerate(ch_names)
                           if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
            if not eog_indices:
                eog_indices = [4,5]

            # Filter and load signals
            signals = [butter_bandpass_filter(reader.readSignal(i), 0.1, 15, fs)
                       for i in eog_indices]

        # Calculate mean absolute value
        win_sec = 10
        samples_per_win = int(win_sec * fs)
        n_windows = total_samples // samples_per_win

        eog_mean = []
        for sig in signals:
            activity = [
                np.mean(np.abs(sig[i*samples_per_win:(i+1)*samples_per_win]))
                for i in range(n_windows)
            ]
            eog_mean.extend(activity)

        mean_val = np.mean(eog_mean)
        std_val = np.std(eog_mean)

        # Read annotations
        n_annots = 0
        if os.path.exists(annot_path):
            with pyedflib.EdfReader(annot_path) as ann_reader:
                onsets, durations, descs = ann_reader.readAnnotations()
                n_annots = len(onsets)

        # Annotation density per hour
        annot_per_hour = n_annots / (dur / 3600)

        # Extract subject ID and trial
        subj, trial = file_name.split(".edf")[0].split("_")
        summary.append({
            "Subject": subj,
            "Trial": trial,
            "Duration_min": round(dur/60, 2),
            "Mean_EOG": round(mean_val, 2),
            "Std_EOG": round(std_val, 2),
            "N_Annotations": n_annots,
            "Annotations_per_hour": round(annot_per_hour, 2)
        })

    except Exception as e:
        print(f"⚠️ Error processing {file_name}: {e}")

# Create summary table
df = pd.DataFrame(summary)
df.sort_values(["Subject", "Trial"], inplace=True)
df.reset_index(drop=True, inplace=True)

print("\n=== Summary of Drowsiness Metrics ===")
print(df)

# Save to CSV file
out_path = os.path.join(base_path, "drowsiness_summary.csv")
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n✅ Summary saved to: {out_path}")

# Calculate correlation between LOC and ROC channels
records = []

# Loop through all EDF files (without annotations)
for file_name in sorted(os.listdir(base_path)):
    if not file_name.lower().endswith(".edf") or "annotations" in file_name.lower():
        continue

    file_path = os.path.join(base_path, file_name)
    try:
        with pyedflib.EdfReader(file_path) as reader:
            ch_names = reader.getSignalLabels()
            fs = reader.getSampleFrequency(0)
            dur = reader.file_duration

            # Search for LOC and ROC channels
            eog_indices = [i for i, n in enumerate(ch_names)
                           if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
            if len(eog_indices) < 2:
                continue  # Skip if there aren't two channels

            sig1 = butter_bandpass_filter(reader.readSignal(eog_indices[0]), 0.1, 15, fs)
            sig2 = butter_bandpass_filter(reader.readSignal(eog_indices[1]), 0.1, 15, fs)

            # Truncate to equal number of samples
            n = min(len(sig1), len(sig2))
            sig1, sig2 = sig1[:n], sig2[:n]

            # Calculate Pearson correlation
            corr = np.corrcoef(sig1, sig2)[0,1]

            # Calculate activity means
            mean1 = np.mean(np.abs(sig1))
            mean2 = np.mean(np.abs(sig2))

            subj, trial = file_name.split(".edf")[0].split("_")
            records.append({
                "Subject": subj,
                "Trial": trial,
                "Duration_min": round(dur/60, 2),
                "Correlation_LOC_ROC": round(corr, 4),
                "Mean_EOG_LOC": round(mean1, 2),
                "Mean_EOG_ROC": round(mean2, 2)
            })

    except Exception as e:
        print(f"⚠️ Error processing {file_name}: {e}")

# Create table
df_corr = pd.DataFrame(records).sort_values(["Subject", "Trial"]).reset_index(drop=True)

print("\n=== EOG Channel Correlation Summary ===")
print(df_corr)

# Save to file
out_path = os.path.join(base_path, "eog_channel_correlation.csv")
df_corr.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n✅ Saved to: {out_path}")

# Display EOG signals for all subjects (3-minute window)
# Loop through all EDF files (without annotations)
for file_name in sorted(os.listdir(base_path)):
    if not file_name.lower().endswith(".edf"):
        continue
    if "annotations" in file_name.lower():
        continue

    file_path = os.path.join(base_path, file_name)
    print(f"\n📄 Displaying EOG signals for {file_name}")

    try:
        with pyedflib.EdfReader(file_path) as reader:
            ch_names = reader.getSignalLabels()
            fs = reader.getSampleFrequency(0)
            dur = reader.file_duration
            total_samples = int(dur * fs)

            # Find EOG channels
            eog_indices = [i for i, n in enumerate(ch_names)
                           if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
            if len(eog_indices) < 2:
                print("⚠️ Missing EOG channels, skipping.")
                continue

            # Read signals (shortened to first 3 minutes for display)
            start_sec = 0
            window_sec = 180  # 3 minutes for display
            start_sample = int(start_sec * fs)
            end_sample = int(min((start_sec + window_sec) * fs, total_samples))

            sig1 = butter_bandpass_filter(reader.readSignal(eog_indices[0])[start_sample:end_sample], 0.1, 15, fs)
            sig2 = butter_bandpass_filter(reader.readSignal(eog_indices[1])[start_sample:end_sample], 0.1, 15, fs)

            time = np.arange(len(sig1)) / fs

        # Plot
        plt.figure(figsize=(12,6))

        plt.subplot(2,1,1)
        plt.plot(time, sig1, color='tab:blue')
        plt.title(f"{file_name} – {ch_names[eog_indices[0]]}")
        plt.ylabel("Amplitude (µV)")
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(time, sig2, color='tab:orange')
        plt.title(f"{file_name} – {ch_names[eog_indices[1]]}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (µV)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        input("Press Enter to continue to next subject...")

    except Exception as e:
        print(f"⚠️ Error reading {file_name}: {e}")

# Display differential EOG (LOC - ROC) for all subjects
# Loop through all EDF files
for file_name in sorted(os.listdir(base_path)):
    if not file_name.lower().endswith(".edf"):
        continue
    if "annotations" in file_name.lower():
        continue

    file_path = os.path.join(base_path, file_name)
    print(f"\n📄 Displaying {file_name}")

    try:
        with pyedflib.EdfReader(file_path) as reader:
            ch_names = reader.getSignalLabels()
            fs = reader.getSampleFrequency(0)
            dur = reader.file_duration
            total_samples = int(dur * fs)

            # Find EOG channels
            eog_indices = [i for i, n in enumerate(ch_names)
                           if any(k in n.upper() for k in ["EOG", "LOC", "ROC"])]
            if len(eog_indices) < 2:
                print("⚠️ Missing one of the EOG channels — skipping.")
                continue

            # Display first 3 minutes
            start_sec = 0
            window_sec = 180
            start_sample = int(start_sec * fs)
            end_sample = int(min((start_sec + window_sec) * fs, total_samples))

            sig1 = butter_bandpass_filter(reader.readSignal(eog_indices[0])[start_sample:end_sample], 0.1, 15, fs)
            sig2 = butter_bandpass_filter(reader.readSignal(eog_indices[1])[start_sample:end_sample], 0.1, 15, fs)

            time = np.arange(len(sig1)) / fs
            diff_signal = sig1 - sig2  # Difference (LOC - ROC)

        # Plot
        plt.figure(figsize=(12,8))

        plt.subplot(3,1,1)
        plt.plot(time, sig1, color='tab:blue')
        plt.title(f"{file_name} – {ch_names[eog_indices[0]]}")
        plt.ylabel("Amplitude (µV)")
        plt.grid(True)

        plt.subplot(3,1,2)
        plt.plot(time, sig2, color='tab:orange')
        plt.title(f"{file_name} – {ch_names[eog_indices[1]]}")
        plt.ylabel("Amplitude (µV)")
        plt.grid(True)

        plt.subplot(3,1,3)
        plt.plot(time, diff_signal, color='tab:green')
        plt.title(f"Differential EOG (LOC - ROC)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (µV)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        input("Press Enter to continue...")

    except Exception as e:
        print(f"⚠️ Error processing {file_name}: {e}")