# %%
"""
Data Exploration for Drowsiness Detection

This script provides a quick overview of the dataset for Google Colab.
For comprehensive EDA, see ../preprocessing/eda.py

Key Points:
- Dataset overview
- Labeling methodology
- Class imbalance
- Train/val/test split rationale
"""

# %%
# Setup
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100

print("=" * 80)
print("DROWSINESS DETECTION - DATA EXPLORATION")
print("=" * 80)

# %%
# Dataset Overview
print("\n📊 DATASET OVERVIEW")
print("-" * 80)
print("Source: Dryad Drowsiness Dataset")
print("URL: https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c")
print("\nCharacteristics:")
print("  • Subjects: 10 participants")
print("  • Sessions: 20 recording sessions (2 per subject)")
print("  • Drowsiness Events: 816 annotated episodes")
print("  • Channels: 2 EOG channels (LOC-Ref, ROC-Ref)")
print("  • Original Sampling Rate: 128 Hz")
print("  • Resampled to: 200 Hz (for CNN compatibility)")
print("  • Original Format: EDF files")
print("  • Processed Format: .mat files")

# %%
# Labeling Methodology
print("\n📋 LABELING METHODOLOGY")
print("-" * 80)
print("2-SECOND CONTEXT WINDOW APPROACH:")
print("  1. Original annotation marks exact drowsiness event")
print("  2. Context extension: ±1 second around each event")
print("  3. Total window: 2 seconds")
print("  4. Samples in window → labeled as 'Drowsy' (1)")
print("  5. All other samples → labeled as 'Awake' (0)")
print("\nRationale:")
print("  ✓ Captures gradual transitions into/out of drowsiness")
print("  ✓ Provides more training data")
print("  ✓ Maintains clinical relevance")
print("  ✓ Accounts for annotation imprecision")

# %%
# Class Imbalance
print("\n⚠️  CLASS IMBALANCE")
print("-" * 80)
print("Severe imbalance in the dataset:")
print("  • Awake samples:  ~96.5%")
print("  • Drowsy samples: ~3.5%")
print("  • Imbalance ratio: ~27:1 (Awake:Drowsy)")
print("\nHandling Strategy:")
print("  • Class weighting: Drowsy samples weighted 30-40x more")
print("  • Sample weights in data generator")
print("  • Cohen's Kappa as primary metric (handles imbalance)")
print("  • Precision-Recall analysis (not just accuracy)")

# Visualize class imbalance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
colors = ['#2ecc71', '#e74c3c']
ax1.pie([96.5, 3.5], 
        labels=['Awake', 'Drowsy'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=(0, 0.1))
ax1.set_title('Class Distribution\n(Severe Imbalance)', fontsize=14, fontweight='bold')

# Bar chart (log scale)
awake_samples = 35000000  # Approximate
drowsy_samples = 1200000  # Approximate
ax2.bar(['Awake', 'Drowsy'], 
        [awake_samples, drowsy_samples],
        color=colors,
        edgecolor='black',
        linewidth=2)
ax2.set_ylabel('Number of Samples', fontsize=12)
ax2.set_title('Sample Count by Class\n(Log Scale)', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (label, v) in enumerate([('Awake', awake_samples), ('Drowsy', drowsy_samples)]):
    ax2.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print("\n💡 Key Takeaway:")
print("   Standard accuracy is misleading with this imbalance!")
print("   A model predicting all 'Awake' would get 96.5% accuracy but be useless.")

# %%
# Subject-Wise Distribution
print("\n📈 SUBJECT-WISE DROWSINESS DISTRIBUTION")
print("-" * 80)

# Example data (from actual analysis)
subjects = ['07F', '01M', '02F', '06M', '03F', '08M', '10M', '04M', '05M', '09M']
event_counts = [100, 95, 88, 82, 78, 75, 70, 65, 60, 55]  # Approximate

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(subjects, event_counts, color='coral', edgecolor='black', linewidth=1.5)

# Highlight Subject 07
bars[0].set_color('#e74c3c')
bars[0].set_linewidth(3)

ax.set_xlabel('Number of Drowsiness Events', fontsize=12)
ax.set_ylabel('Subject', fontsize=12)
ax.set_title('Drowsiness Events per Subject', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (subj, count) in enumerate(zip(subjects, event_counts)):
    ax.text(count + 2, i, str(count), va='center', fontsize=10)

# Add annotation for Subject 07
ax.text(event_counts[0] + 10, 0, '← TEST SET', 
        va='center', fontsize=11, fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.show()

print("\n🎯 Subject 07 chosen as TEST SET:")
print("   • Has the MOST drowsiness events (~100,600 samples)")
print("   • Provides robust evaluation dataset")
print("   • Ensures fair comparison between models")

# %%
# Train/Val/Test Split Strategy
print("\n🔀 TRAIN/VAL/TEST SPLIT STRATEGY")
print("-" * 80)
print("SUBJECT-BASED SPLITTING (prevents data leakage):")
print("\n  Training Set:")
print("    • Subjects: 01-06")
print("    • Files: 12 (2 sessions per subject)")
print("    • Samples: ~35M total")
print("    • Purpose: Model training")
print("\n  Validation Set:")
print("    • Subjects: 08-10")
print("    • Files: 6 (2 sessions per subject)")
print("    • Samples: ~9M total")
print("    • Purpose: Hyperparameter tuning, early stopping")
print("\n  Test Set:")
print("    • Subject: 07")
print("    • Files: 2 (2 sessions)")
print("    • Samples: ~100,600 drowsy samples")
print("    • Purpose: Final evaluation, model comparison")
print("\nWhy subject-based splitting?")
print("  ✓ Prevents data leakage (no subject in multiple sets)")
print("  ✓ Tests generalization to new subjects")
print("  ✓ More realistic evaluation")
print("  ✓ Subject 07 has most drowsy samples → robust test set")

# Visualize split
fig, ax = plt.subplots(figsize=(10, 6))

split_data = {
    'Training\n(Subjects 01-06)': 12,
    'Validation\n(Subjects 08-10)': 6,
    'Test\n(Subject 07)': 2
}

colors_split = ['#3498db', '#f39c12', '#e74c3c']
bars = ax.bar(split_data.keys(), split_data.values(), 
              color=colors_split, edgecolor='black', linewidth=2)

ax.set_ylabel('Number of Files', fontsize=12)
ax.set_title('Train/Val/Test Split Distribution', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)} files',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Signal Characteristics
print("\n📡 SIGNAL CHARACTERISTICS")
print("-" * 80)
print("EOG Channels:")
print("  • LOC-Ref: Left Outer Canthus referenced")
print("  • ROC-Ref: Right Outer Canthus referenced")
print("\nSignal Processing:")
print("  • Bandpass filter: 0.1-10 Hz (removes DC offset and noise)")
print("  • Resampling: 128 Hz → 200 Hz (anti-aliasing applied)")
print("  • Normalization: Per-channel z-score normalization")
print("\nWindow Sizes Tested:")
print("  • CNN_2s:  400 samples (2 sec × 200 Hz)")
print("  • CNN_4s:  800 samples (4 sec)")
print("  • CNN_8s:  1,600 samples (8 sec)")
print("  • CNN_16s: 3,200 samples (16 sec) ⭐ BEST")
print("  • CNN_32s: 6,400 samples (32 sec)")

# %%
# Key Takeaways
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("\n1. SEVERE CLASS IMBALANCE:")
print("   → Requires class weighting and appropriate metrics (Kappa)")
print("\n2. SUBJECT-BASED SPLIT:")
print("   → Prevents data leakage, tests generalization")
print("\n3. SUBJECT 07 AS TEST SET:")
print("   → Most drowsy samples, robust evaluation")
print("\n4. 2-SECOND CONTEXT WINDOWS:")
print("   → Captures transitions, provides more training data")
print("\n5. 16-SECOND WINDOW OPTIMAL:")
print("   → Best balance between temporal context and performance")
print("\n" + "=" * 80)
print("END OF DATA EXPLORATION")
print("=" * 80)

# %%

