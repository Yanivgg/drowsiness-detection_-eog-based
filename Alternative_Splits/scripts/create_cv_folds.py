"""
Create Leave-One-Subject-Out Cross-Validation Folds

Creates 10 fold files, each with one subject as test and the other 9 as train.

Fold 01: test=Subject 01, train=02-10
Fold 02: test=Subject 02, train=01,03-10
...
Fold 10: test=Subject 10, train=01-09
"""

import scipy.io as spio
import numpy as np
import os

print("="*80)
print("Creating Leave-One-Subject-Out CV Folds")
print("="*80)

# All subjects with their recordings
all_subjects = ['01M', '02F', '03F', '04M', '05M', '06M', '07F', '08M', '09M', '10M']

# Create 10 folds
for fold_idx, test_subj in enumerate(all_subjects, 1):
    print(f"\nCreating Fold {fold_idx:02d} (Test: {test_subj})...")
    
    # Get all files for test subject (both _1 and _2)
    test_files = [f'{test_subj}_1', f'{test_subj}_2']
    
    # Get all files for other subjects (train)
    train_files = []
    for subj in all_subjects:
        if subj != test_subj:
            train_files.extend([f'{subj}_1', f'{subj}_2'])
    
    # Create file_sets dictionary
    file_sets = {
        'train_files': np.array(train_files, dtype=object),
        'val_files': np.array([], dtype=object),  # No validation set
        'test_files': np.array(test_files, dtype=object)
    }
    
    # Save fold
    output_path = f'../cross_validation/fold_{fold_idx:02d}.mat'
    spio.savemat(output_path, file_sets)
    
    print(f"  ✓ Saved: fold_{fold_idx:02d}.mat")
    print(f"    Train: {len(train_files)} files (9 subjects)")
    print(f"    Test:  {len(test_files)} files ({test_subj})")

print("\n" + "="*80)
print(f"Created {len(all_subjects)} CV folds successfully!")
print("="*80)
print("\nFold Summary:")
for i, subj in enumerate(all_subjects, 1):
    print(f"  Fold {i:02d}: Test={subj}, Train=other 9 subjects")
print("\nUsage:")
print("  Run train_cv_colab.py to train all 10 folds")
print("="*80)

