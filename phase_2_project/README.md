# Phase 2: Feature-Based Drowsiness Detection

## Overview

This is Phase 2 of the drowsiness detection project, using **feature engineering + traditional ML classifiers** instead of deep learning (CNN).

### Key Information:
- **Goal**: Detect drowsiness from EOG signals using engineered features
- **Window**: 16 seconds (3200 samples @ 200 Hz) - matches CNN_16s from Phase 1
- **Stride**: 200 samples (1 second) for sliding window
- **Features**: ~70 features (time-domain, frequency-domain, non-linear, EOG-specific)
- **Models**: Random Forest & SVM
- **Test Set**: Subject 07 (same as Phase 1 for fair comparison)
- **Train Set**: Subjects 01-06, 08-10 (16 files, no validation set)

---

## Workflow

### Step 1: Feature Extraction (Local - LONG RUNTIME)

Extract features from original EDF files:

```bash
python phase_2_project/feature_extraction.py
```

**What it does:**
- Reads EDF files from `original_data/`
- Applies bandpass filter (0.1-10 Hz)
- Resamples to 200 Hz
- Creates 16s sliding windows (stride=1s)
- Extracts ~70 features per window
- Saves CSVs to `phase_2_project/features/`

**Runtime**: ~30-60 minutes (depending on CPU)

**Output:**
- `phase_2_project/features/train_features_16s.csv` (~150 MB)
- `phase_2_project/features/test_features_16s.csv` (~20 MB)

---

### Step 2: Train ML Models (Google Colab - GPU)

1. **Upload** `train_ml_colab.py` to Google Colab
2. **Convert** to notebook (or copy cell-by-cell)
3. **Upload** the two CSV files when prompted:
   - `train_features_16s.csv`
   - `test_features_16s.csv`
4. **Run** all cells

**What it does:**
- Loads feature CSVs
- Standardizes features
- Trains Random Forest (300 trees, class-balanced)
- Trains SVM (RBF kernel, class-balanced)
- Evaluates on Subject 07
- Compares with CNN_16s results
- Saves models and plots to Google Drive

**Runtime**: ~10-20 minutes

**Output** (saved to Google Drive):
- `rf_model.pkl`, `svm_model.pkl`, `scaler.pkl`
- `feature_importance.csv`
- `model_comparison.csv`
- Confusion matrices (PNG)
- Comparison plots (PNG)

---

## Feature Categories

### Time-Domain (26 features)
- Basic stats: mean, std, var, min, max, range, median
- Shape: peak/valley amplitude, rise/fall time, inflections
- Moments: skewness, kurtosis
- Signal: zero-crossing rate, RMS, crest factor, energy
- Hjorth: activity, mobility, complexity

### Frequency-Domain (20 features)
- Band powers: delta, theta, alpha, beta (absolute + relative)
- Spectral: centroid, spread, entropy, edge frequency
- Ratios: theta/alpha, delta/beta, alpha/beta, low/high

### Non-Linear (10 features)
- Entropy: sample entropy, approximate entropy, permutation entropy
- Fractal: Higuchi, Petrosian, Katz, Hurst exponent
- Complexity: Lempel-Ziv, DFA alpha, mobility, complexity

### EOG-Specific (14 features)
- Blinks: rate, amplitude (mean, std)
- Slow eye movements: count, duration
- LOC-ROC: correlation, lag, lag correlation
- Movements: amplitude, velocity (mean, std), saccade rate
- Asymmetry: energy ratio, amplitude ratio

---

## Expected Results

### Random Forest (Primary Model)
- **Kappa**: 0.25-0.35
- **Recall**: 20-30%
- **Precision**: 40-55%

### SVM (Secondary Model)
- **Kappa**: 0.20-0.30
- **Recall**: 15-25%
- **Precision**: 35-50%

### Comparison with CNN_16s (Phase 1)
- **CNN_16s Kappa**: 0.394
- **CNN_16s Recall**: 32.7%
- **CNN_16s Precision**: 54.9%

**Why ML is lower?**
- Engineered features vs. learned representations
- Less data augmentation (stride 200 vs 1)
- Simpler model architecture

**Advantages of ML:**
- Interpretable features
- Feature importance analysis
- No GPU needed for inference
- Faster training
- Lower computational cost

---

## Files Structure

```
phase_2_project/
├── README.md                          # This file
├── feature_extraction.py              # LOCAL: Extract features from EDF
├── train_ml_colab.py                  # COLAB: Train RF & SVM
├── requirements_phase2.txt            # Python dependencies
├── feature_engineering/               # Feature extraction modules
│   ├── __init__.py
│   ├── time_domain.py                 # Time-domain features
│   ├── frequency_domain.py            # Frequency-domain features
│   ├── nonlinear.py                   # Non-linear features
│   └── eog_specific.py                # EOG-specific features
├── features/                          # OUTPUT: Feature CSVs
│   ├── train_features_16s.csv         # Train set (Subjects 01-06,08-10)
│   └── test_features_16s.csv          # Test set (Subject 07)
└── old/                               # Old/reference code
```

---

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn tqdm mne matplotlib seaborn
```

Or use:
```bash
pip install -r phase_2_project/requirements_phase2.txt
```

---

## Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| **Test Set** | Subject 07 | Same as CNN for fair comparison, 100K+ drowsy samples |
| **Train Set** | 01-06, 08-10 | More data for ML (16 vs 12 files) |
| **Window** | 16s | Match CNN_16s (best Phase 1 model) |
| **Stride** | 200 samples (1s) | Balance between data size and augmentation |
| **Features** | ~70 | Comprehensive coverage of all domains |
| **Models** | RF + SVM | Interpretable, proven, class-balanced |
| **No Validation** | Train-Test only | ML uses intrinsic validation, simpler workflow |

---

## Comparison with Phase 1

| Aspect | Phase 1 (CNN) | Phase 2 (ML) |
|--------|---------------|--------------|
| Test Set | Subject 07 | Subject 07 ✅ |
| Window | 16s | 16s ✅ |
| Train Data | 01-06 (12 files) | 01-06,08-10 (16 files) |
| Stride | 1 sample | 200 samples |
| Input | Raw EOG (2 channels) | 70 engineered features |
| Approach | End-to-end CNN | Feature extraction + RF/SVM |
| Training | GPU, 30-90 min | CPU/GPU, 10-20 min |
| Interpretability | Low (black box) | High (feature importance) |
| Expected Kappa | 0.39 | 0.25-0.35 |

---

## Troubleshooting

### Feature Extraction Issues

**Problem**: "No module named 'mne'"
```bash
pip install mne
```

**Problem**: Long runtime
- **Solution**: Run on a powerful CPU, or reduce stride (e.g., stride=400 for 2s overlap)

**Problem**: Memory error
- **Solution**: Process files in batches, or increase system RAM

### Training Issues

**Problem**: "NaN in features"
- **Solution**: Already handled by `np.nan_to_num` in training script

**Problem**: Low performance
- **Causes**: Class imbalance (97% awake, 3% drowsy)
- **Solution**: Already using `class_weight='balanced'`

---

## Future Improvements

1. **Feature Selection**: Use feature importance to reduce dimensionality
2. **Hyperparameter Tuning**: Grid search for RF & SVM parameters
3. **More Models**: Try XGBoost, LightGBM
4. **Different Windows**: Try 8s or 32s windows
5. **Ensemble**: Combine RF + SVM predictions
6. **SMOTE**: Oversample minority class for training

---

## Citation

If using this code, please cite the original Dryad dataset:

```
Goolkate, Remco; Ijsselsteijn, Wijnand (2024). 
Monotonous Task EEG and EOG data [Dataset]. 
Dryad. https://doi.org/10.5061/dryad.6hdr7sqz4
```

---

## Contact

For questions or issues, please refer to the main project README.

**Author**: Yaniv Grosberg  
**Date**: November 2025  
**Phase**: 2 of 2 (Feature Engineering)

