"""
Create Within-Subject Split

Train: All _1 recordings (10 files)
Test: All _2 recordings (10 files)

This tests temporal generalization - can the model trained on first recordings
generalize to second recordings of the same subjects?
"""

import scipy.io as spio
import numpy as np
import os

print("="*80)
print("Creating Within-Subject Split")
print("="*80)

# All subjects with their recordings
all_subjects = ['01M', '02F', '03F', '04M', '05M', '06M', '07F', '08M', '09M', '10M']

# Within-Subject Split: _1 for train, _2 for test
train_files = [f'{subj}_1' for subj in all_subjects]
test_files = [f'{subj}_2' for subj in all_subjects]

print(f"\nTrain files ({len(train_files)}):")
for f in train_files:
    print(f"  - {f}")

print(f"\nTest files ({len(test_files)}):")
for f in test_files:
    print(f"  - {f}")

# Create file_sets dictionary
file_sets = {
    'train_files': np.array(train_files, dtype=object),
    'val_files': np.array([], dtype=object),  # No validation set
    'test_files': np.array(test_files, dtype=object)
}

# Save to .mat file
output_path = '../within_subject/file_sets.mat'
spio.savemat(output_path, file_sets)

print(f"\n✓ Saved: {output_path}")
print(f"\nSummary:")
print(f"  Train: {len(train_files)} files (all _1 recordings)")
print(f"  Val:   0 files")
print(f"  Test:  {len(test_files)} files (all _2 recordings)")
print("\n" + "="*80)
print("Within-Subject split created successfully!")
print("="*80)

