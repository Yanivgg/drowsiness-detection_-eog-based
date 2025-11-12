# Data Preprocessing

Scripts for converting raw EDF files to processed MAT format suitable for deep learning.

## Files

### `edf_to_mat_eog_only.py`
Main preprocessing script that converts EDF recordings to MAT format.

**Input**:
- `original_data/*.edf` - Raw EDF recordings
- `original_data/*_annotations.edf` - Annotation files

**Output**:
- `data/processed/files/*.mat` - Processed MAT files (20 files total)

**Usage**:
```bash
python edf_to_mat_eog_only.py
```

**What it does**:
1. Loads EDF files with EOG signals
2. Extracts LOC-Ref and ROC-Ref channels (E1, E2)
3. Resamples from 128 Hz to 200 Hz
4. Loads microsleep annotations
5. Creates binary labels (0=Awake, 1=Drowsy)
6. Applies 2-second window around events
7. Saves as MAT format

**Output structure**:
```matlab
Data
├── E1           % EOG channel 1 (n_samples,)
├── E2           % EOG channel 2 (n_samples,)
├── labels_O1    % Binary labels (n_samples,)
└── labels_O2    % Binary labels (n_samples,)
```

### `eda.py`
Exploratory Data Analysis script.

**What it does**:
- Analyzes EDF file structure
- Counts annotations per subject
- Calculates event statistics
- Generates summary CSV files

**Output**:
- `annotations_summary.csv` - Event counts per subject
- `dryad_eda_summary.csv` - Dataset statistics

### `annotations_summary.csv`
Event distribution across subjects and sessions.

**Columns**:
- Subject ID
- Session number
- Microsleep event count
- Total recording duration
- Events per minute

### `dryad_eda_summary.csv`
Dataset metadata and statistics.

**Columns**:
- Subject demographics
- Channel information
- Sampling rates
- Recording durations

## Quick Start

**Prerequisites**:
```bash
pip install mne pyedflib numpy scipy
```

**Run preprocessing**:
```bash
# 1. Place raw EDF files in original_data/
# 2. Run conversion
python edf_to_mat_eog_only.py

# 3. Verify output
ls ../data/processed/files/  # Should see 20 .mat files
```

## Output Verification

**Check processed file**:
```python
import scipy.io as spio

data = spio.loadmat('../data/processed/files/01M_1.mat')
e1 = data['Data']['E1'][0,0].flatten()
labels = data['Data']['labels_O1'][0,0].flatten()

print(f"Samples: {len(e1):,}")
print(f"Drowsy: {labels.sum():,} ({labels.mean()*100:.2f}%)")
print(f"Shape: E1={e1.shape}, E2={data['Data']['E2'][0,0].shape}")
```

**Expected**:
- Samples: ~1,400,000 per file
- Drowsy: 3-5%
- Shape: (n_samples,) for each channel

## Troubleshooting

**Issue**: Channel names not found  
**Solution**: Check available channels with `mne.io.read_raw_edf(file).ch_names`

**Issue**: Out of memory  
**Solution**: Process files one at a time, reduce resampling buffer size

**Issue**: No annotations found  
**Solution**: Verify annotation EDF file exists and matches expected format

## Next Steps

After preprocessing:
1. Verify data quality (see [PREPROCESSING_GUIDE.md](../docs/PREPROCESSING_GUIDE.md))
2. Create data split: `python ../data/create_file_sets.py`
3. Proceed to training (see [TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md))

## References

- MNE-Python: https://mne.tools/
- pyEDFlib: https://github.com/holgern/pyedflib
- Dryad dataset: https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c

