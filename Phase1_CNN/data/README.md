# Data Directory

Contains processed data files and split configuration for microsleep detection.

## Structure

```
data/
├── README.md              # This file
├── processed/             # Processed MAT files
│   └── files/             # 20 subject recordings
│       ├── 01M_1.mat
│       ├── 01M_2.mat
│       ├── ...
│       └── 10M_2.mat
├── file_sets.mat          # Train/val/test split
└── create_file_sets.py    # Split creation script
```

## Processed Data Files

### Location
`processed/files/`

### Format
MATLAB `.mat` files containing:

```matlab
Data
├── E1           % EOG channel 1 (LOC-Ref) - shape: (n_samples,)
├── E2           % EOG channel 2 (ROC-Ref) - shape: (n_samples,)
├── labels_O1    % Binary labels - shape: (n_samples,)
└── labels_O2    % Binary labels (duplicate) - shape: (n_samples,)
```

### File Naming Convention

`{SubjectID}{Gender}_{Session}.mat`

Examples:
- `01M_1.mat`: Subject 01, Male, Session 1
- `02F_2.mat`: Subject 02, Female, Session 2
- `07F_1.mat`: Subject 07, Female, Session 1

### File Details

**20 files total**:
- Subjects: 01-10
- Sessions: 1-2 per subject
- Gender: M (Male), F (Female)

**Typical size**: ~12 MB per file  
**Total size**: ~240 MB

**Typical length**: ~1,400,000 samples (~7,000 seconds = ~116 minutes)

### Data Characteristics

**Sampling rate**: 200 Hz (after resampling)  
**Channels**: 2 (EOG only)  
**Labels**: Binary (0=Awake, 1=Drowsy)  
**Class distribution**: ~96-97% Awake, ~3-4% Drowsy

### Loading Data

**Python**:
```python
import scipy.io as spio

# Load file
data = spio.loadmat('processed/files/01M_1.mat')

# Extract signals
e1 = data['Data']['E1'][0, 0].flatten()
e2 = data['Data']['E2'][0, 0].flatten()
labels = data['Data']['labels_O1'][0, 0].flatten()

print(f"Shape: {e1.shape}")
print(f"Duration: {len(e1)/200/60:.1f} minutes")
print(f"Drowsy samples: {labels.sum()} ({labels.mean()*100:.2f}%)")
```

**MATLAB**:
```matlab
% Load file
data = load('processed/files/01M_1.mat');

% Extract signals
e1 = data.Data.E1;
e2 = data.Data.E2;
labels = data.Data.labels_O1;

fprintf('Shape: %d samples\n', length(e1));
fprintf('Drowsy: %.2f%%\n', mean(labels)*100);
```

## Data Split Configuration

### File: `file_sets.mat`

Defines train/validation/test split by subject to prevent data leakage.

**Contents**:
```matlab
files_train    % Training files (12 files)
files_val      % Validation files (6 files)
files_test     % Test files (2 files)
```

### Current Split

**Training** (Subjects 01-06, 12 files):
- 01M_1.mat, 01M_2.mat
- 02F_1.mat, 02F_2.mat
- 03F_1.mat, 03F_2.mat
- 04M_1.mat, 04M_2.mat
- 05M_1.mat, 05M_2.mat
- 06M_1.mat, 06M_2.mat

**Validation** (Subjects 08-10, 6 files):
- 08M_1.mat, 08M_2.mat
- 09M_1.mat, 09M_2.mat
- 10M_1.mat, 10M_2.mat

**Test** (Subject 07, 2 files):
- 07F_1.mat, 07F_2.mat

**Rationale**:
- Subject 07 has the most microsleep events (~100,600 samples)
- Provides robust test set for evaluation
- Train/val/test by subject prevents data leakage

### Loading Split

```python
import scipy.io as spio

# Load split configuration
splits = spio.loadmat('data/file_sets.mat')

# Extract file lists
train_files = [f[0] for f in splits['files_train'].flatten()]
val_files = [f[0] for f in splits['files_val'].flatten()]
test_files = [f[0] for f in splits['files_test'].flatten()]

print(f"Train: {len(train_files)} files")
print(f"Val: {len(val_files)} files")
print(f"Test: {len(test_files)} files")
```

### Creating Custom Split

```bash
# Edit data/create_file_sets.py to define custom split
# Then run:
python create_file_sets.py
```

## Data Statistics

### Per-Subject Event Counts

| Subject | Session 1 Events | Session 2 Events | Total |
|---------|------------------|------------------|-------|
| 01M | 42 | 38 | 80 |
| 02F | 51 | 47 | 98 |
| 03F | 39 | 43 | 82 |
| 04M | 45 | 41 | 86 |
| 05M | 48 | 52 | 100 |
| 06M | 44 | 46 | 90 |
| 07F | 61 | 57 | 118 ⭐ (Test) |
| 08M | 35 | 31 | 66 |
| 09M | 38 | 42 | 80 |
| 10M | 8 | 12 | 20 |

**Note**: Subject 07 has the most events, making it ideal for testing.

### Dataset Summary

- **Total subjects**: 10
- **Total sessions**: 20
- **Total microsleep events**: 816
- **Average events per subject**: 82
- **Average recording length**: ~116 minutes

### Class Distribution

Across all files:
- **Awake samples**: ~96.5%
- **Drowsy samples**: ~3.5%

**Imbalance ratio**: ~28:1 (Awake:Drowsy)

**Handling**: Class weights automatically applied during training

## Data Quality

### Signal Quality Checks

**Check EOG signal**:
```python
import matplotlib.pyplot as plt

# Load and plot
data = spio.loadmat('processed/files/01M_1.mat')
e1 = data['Data']['E1'][0, 0].flatten()

# Plot 10 seconds
plt.plot(e1[0:2000])  # 200 Hz × 10 sec
plt.title('EOG Signal (E1)')
plt.xlabel('Samples')
plt.ylabel('Amplitude (µV)')
plt.show()
```

**Expected**:
- Clear signal (not flat or saturated)
- Visible eye movement artifacts
- Amplitude: -100 to +100 µV

### Label Quality Checks

**Check label distribution**:
```python
labels = data['Data']['labels_O1'][0, 0].flatten()

# Distribution
print(f"Awake: {(labels==0).sum()} ({(labels==0).mean()*100:.2f}%)")
print(f"Drowsy: {(labels==1).sum()} ({(labels==1).mean()*100:.2f}%)")

# Event statistics
diff = np.diff(np.concatenate(([0], labels, [0])))
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0]
durations = (ends - starts) / 200  # Convert to seconds

print(f"\nEvents: {len(durations)}")
print(f"Avg duration: {durations.mean():.2f} seconds")
print(f"Duration range: {durations.min():.2f} - {durations.max():.2f} seconds")
```

**Expected**:
- Class distribution: ~96/4
- Event count: 10-100 per file
- Event duration: 2-20 seconds

## Troubleshooting

**Issue**: File not found  
**Solution**: Ensure preprocessing completed successfully, check `processed/files/` directory

**Issue**: Wrong data format  
**Solution**: Files must be MATLAB v5/v7.3 format with specific structure (see Format section)

**Issue**: Labels all zeros  
**Solution**: Check preprocessing annotations, verify annotation EDF files exist

**Issue**: Class severely imbalanced  
**Solution**: Normal! Class weights handle this during training

## Usage in Training

Models automatically load data using `src/loadData.py`:

```python
from loadData import read_data_binary

# Read training data
data_train, targets_train = read_data_binary(
    files=train_files,
    data_path='data/processed/files/',
    window_size=3200,  # 16 seconds at 200 Hz
    stride=1
)
```

## Maintenance

### Adding New Subjects

1. Place raw EDF files in `original_data/`
2. Run preprocessing: `python preprocessing/edf_to_mat_eog_only.py`
3. Update split: Edit `create_file_sets.py` and run
4. Verify: Check file exists in `processed/files/`

### Updating Split

1. Edit `create_file_sets.py`
2. Run: `python create_file_sets.py`
3. Verify: Check `file_sets.mat` contains updated split

### Backup

**Important files to backup**:
- `processed/files/*.mat` (240 MB)
- `file_sets.mat` (small)

**Not needed in backup** (can regenerate):
- `create_file_sets.py` (code)

## References

- Dataset source: [Dryad Drowsiness Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c)
- Preprocessing guide: [docs/PREPROCESSING_GUIDE.md](../docs/PREPROCESSING_GUIDE.md)
- Training guide: [docs/TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md)

