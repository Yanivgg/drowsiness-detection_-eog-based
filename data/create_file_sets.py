"""
Create file_sets.mat for train/val/test split
Splits by subject to avoid data leakage:
- Train: subjects 01-06 (12 files)
- Validation: subjects 08-10 (6 files) 
- Test: subject 07 (2 files) - Has the most microsleep events (100,600 samples)
"""

import scipy.io as spio
import numpy as np

# Define splits by subject
files_train = [
    '01M_1.mat', '01M_2.mat',
    '02F_1.mat', '02F_2.mat',
    '03F_1.mat', '03F_2.mat',
    '04M_1.mat', '04M_2.mat',
    '05M_1.mat', '05M_2.mat',
    '06M_1.mat', '06M_2.mat'
]

files_val = [
    '08M_1.mat', '08M_2.mat',
    '09M_1.mat', '09M_2.mat',
    '10M_1.mat', '10M_2.mat'
]

files_test = [
    '07F_1.mat', '07F_2.mat'
]

# Convert to numpy arrays of objects (MATLAB cell array format)
files_train_arr = np.array([[f] for f in files_train], dtype=object)
files_val_arr = np.array([[f] for f in files_val], dtype=object)
files_test_arr = np.array([[f] for f in files_test], dtype=object)

# Save to .mat file
spio.savemat('file_sets.mat', {
    'files_train': files_train_arr,
    'files_val': files_val_arr,
    'files_test': files_test_arr
})

print("Created file_sets.mat successfully!")
print(f"Train: {len(files_train)} files (subjects 01-06)")
print(f"Validation: {len(files_val)} files (subjects 08-10)")
print(f"Test: {len(files_test)} files (subject 07 - most microsleep events)")
print(f"Total: {len(files_train) + len(files_val) + len(files_test)} files")

