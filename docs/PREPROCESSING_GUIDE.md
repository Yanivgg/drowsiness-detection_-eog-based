# Data Preprocessing Guide

Complete guide for converting raw EDF files to processed MAT format for microsleep detection.

## Overview

The preprocessing pipeline converts raw EDF recordings and annotations into processed `.mat` files suitable for deep learning model training.

**Input**: EDF files with EOG signals and annotation files  
**Output**: MAT files with resampled signals and binary labels

## Prerequisites

```bash
pip install mne pyedflib numpy scipy
```

## Step-by-Step Process

### 1. Prepare Raw Data

**Required files**:
- `{subject}_1.edf` - Recording session 1
- `{subject}_1_annotations.edf` - Annotations for session 1
- `{subject}_2.edf` - Recording session 2
- `{subject}_2_annotations.edf` - Annotations for session 2

Example: `01M_1.edf`, `01M_1_annotations.edf`, `01M_2.edf`, `01M_2_annotations.edf`

**Place files in**: `original_data/` directory

### 2. Run Preprocessing Script

```bash
cd preprocessing
python edf_to_mat_eog_only.py
```

### 3. What the Script Does

#### Step 1: Load EDF Files
```python
# Read EDF file
raw = mne.io.read_raw_edf(edf_path, preload=True)
```

#### Step 2: Extract EOG Channels
- **LOC-Ref**: Left Outer Canthus (E1)
- **ROC-Ref**: Right Outer Canthus (E2)

```python
eog_channels = ['E1-Ref', 'E2-Ref']  # Channel names may vary
```

#### Step 3: Resample to 200 Hz
Original sampling rate: 128 Hz  
Target sampling rate: 200 Hz (to match original model)

```python
raw.resample(200)  # Resample to 200 Hz
```

**Why 200 Hz?**
- Matches the original Malafeev et al. study
- Good balance between temporal resolution and computational cost
- Captures relevant EOG frequency content (0.5-35 Hz)

#### Step 4: Load Annotations
Read microsleep annotations from annotation EDF:

```python
annotations = mne.read_annotations(annotation_path)
```

**Annotation format**:
- `onset`: Start time of microsleep event (seconds)
- `duration`: Duration of event (typically ~3-15 seconds)
- `description`: Event type (e.g., "Microsleep")

#### Step 5: Create Binary Labels
Generate binary labels with 2-second windows:

```python
window_size = 2.0  # seconds
pre_window = 1.0   # 1 second before
post_window = 1.0  # 1 second after

for annotation in annotations:
    start = annotation['onset'] - pre_window
    end = annotation['onset'] + annotation['duration'] + post_window
    labels[start:end] = 1  # Drowsy
```

**Labeling strategy**:
- `0`: Awake (baseline)
- `1`: Drowsy (within 2-second window of microsleep event)

**Rationale**:
- 2-second context window captures drowsiness onset/offset
- Prevents sharp transitions
- Matches model input window sizes

#### Step 6: Save as MAT File
Save processed data in MATLAB format:

```python
scipy.io.savemat(output_path, {
    'Data': {
        'E1': eog1,           # Shape: (n_samples,)
        'E2': eog2,           # Shape: (n_samples,)
        'labels_O1': labels,  # Shape: (n_samples,)
        'labels_O2': labels   # Shape: (n_samples,)
    }
})
```

**File structure**:
```matlab
Data
├── E1           % EOG channel 1 (LOC-Ref)
├── E2           % EOG channel 2 (ROC-Ref)
├── labels_O1    % Binary labels
└── labels_O2    % Binary labels (duplicate)
```

### 4. Verify Output

**Check processed files**:
```bash
ls data/processed/files/
# Should see: 01M_1.mat, 01M_2.mat, ..., 10M_2.mat (20 files)
```

**Verify file content**:
```python
import scipy.io as spio

data = spio.loadmat('data/processed/files/01M_1.mat')
print(data['Data'].dtype.names)  # ['E1', 'E2', 'labels_O1', 'labels_O2']

e1 = data['Data']['E1'][0,0].flatten()
labels = data['Data']['labels_O1'][0,0].flatten()

print(f"Samples: {len(e1)}")
print(f"Drowsy samples: {labels.sum()} ({labels.mean()*100:.2f}%)")
print(f"Sampling rate: 200 Hz (verified)")
```

**Expected output**:
```
Samples: ~1,461,000 per file (varies by recording length)
Drowsy samples: 3-5% of total
Unique labels: {0, 1}
```

## Output Files

### Processed MAT Files
Location: `data/processed/files/`

**20 files total**:
- `01M_1.mat`, `01M_2.mat` (Subject 01, Male, sessions 1-2)
- `02F_1.mat`, `02F_2.mat` (Subject 02, Female, sessions 1-2)
- ... (continue for subjects 03-10)

**Each file contains**:
- **E1**: EOG channel 1 (n_samples,)
- **E2**: EOG channel 2 (n_samples,)
- **labels_O1**: Binary labels (n_samples,)
- **labels_O2**: Binary labels (n_samples,) [duplicate]

**Typical file size**: ~12 MB per file

### Statistics Files
Location: `preprocessing/`

**annotations_summary.csv**:
- Events per subject/session
- Total microsleep count
- Duration statistics

**dryad_eda_summary.csv**:
- Recording lengths
- Channel information
- Sampling rates

## Data Quality Checks

### 1. Signal Quality
```python
import matplotlib.pyplot as plt

# Load file
data = spio.loadmat('data/processed/files/01M_1.mat')
e1 = data['Data']['E1'][0,0].flatten()

# Plot segment
plt.plot(e1[0:2000])  # First 10 seconds
plt.title('EOG Signal Quality Check')
plt.xlabel('Samples (200 Hz)')
plt.ylabel('Amplitude (µV)')
plt.show()
```

**What to look for**:
- ✓ Clear signal (not flat or saturated)
- ✓ Visible eye movement artifacts
- ✓ Reasonable amplitude range (-100 to +100 µV)
- ✗ Excessive noise or artifacts

### 2. Label Distribution
```python
# Check class balance
unique, counts = np.unique(labels, return_counts=True)
print(f"Class 0 (Awake): {counts[0]:,} ({counts[0]/len(labels)*100:.2f}%)")
print(f"Class 1 (Drowsy): {counts[1]:,} ({counts[1]/len(labels)*100:.2f}%)")
```

**Expected**:
- Class 0 (Awake): ~96-97%
- Class 1 (Drowsy): ~3-4%

### 3. Label Continuity
```python
# Check for isolated labels (potential errors)
diff = np.diff(np.concatenate(([0], labels, [0])))
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0]
durations = ends - starts

print(f"Number of drowsy episodes: {len(durations)}")
print(f"Average duration: {durations.mean() / 200:.2f} seconds")
print(f"Min duration: {durations.min() / 200:.2f} seconds")
```

**Expected**:
- Episodes: 10-100 per file (varies)
- Average duration: 3-15 seconds
- Min duration: ≥ 2 seconds (window size)

## Troubleshooting

### Issue: Channel Names Not Found
**Error**: `ValueError: Channel E1-Ref not found`

**Solution**:
```python
# Check available channels
print(raw.ch_names)

# Update channel names in script
eog_channels = ['LOC-Ref', 'ROC-Ref']  # Adjust based on actual names
```

### Issue: Resampling Fails
**Error**: `Memory error during resampling`

**Solution**:
```python
# Process in chunks
raw.load_data()  # Load first
raw.resample(200)  # Then resample
```

### Issue: No Annotations Found
**Error**: `No microsleep annotations in file`

**Solution**:
- Verify annotation EDF file exists
- Check annotation description matches expected pattern
- Inspect annotation file manually with MNE viewer

### Issue: Output File Empty
**Error**: Zero-length arrays in output

**Solution**:
- Check EDF file integrity
- Verify channels exist
- Ensure sufficient recording duration

## Advanced Configuration

### Custom Labeling Window
Adjust the labeling window size:

```python
# In edf_to_mat_eog_only.py
PRE_WINDOW = 0.5   # 0.5 seconds before
POST_WINDOW = 1.5  # 1.5 seconds after
# Total window: 2 seconds (0.5 + event + 1.5)
```

### Different Sampling Rate
Use a different target sampling rate:

```python
TARGET_FS = 250  # Hz (instead of 200)
raw.resample(TARGET_FS)
```

**Note**: Models are trained for 200 Hz. Using different rates requires retraining.

### Multi-Channel Processing
Extract additional channels (if needed):

```python
# Add more channels
channels_to_extract = ['E1-Ref', 'E2-Ref', 'C3-Ref', 'C4-Ref']

# Save in MAT file
data_dict = {
    'E1': eog1,
    'E2': eog2,
    'C3': eeg_c3,
    'C4': eeg_c4,
    'labels': labels
}
```

## Next Steps

After preprocessing:

1. **Verify data quality** using the checks above
2. **Create data split**: Run `python data/create_file_sets.py`
3. **Proceed to training**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## References

- MNE-Python: https://mne.tools/
- pyEDFlib: https://github.com/holgern/pyedflib
- Original dataset: https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c

