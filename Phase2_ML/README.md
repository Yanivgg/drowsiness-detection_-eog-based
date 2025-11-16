# Experiment 2: Traditional ML with Feature Engineering

> Part of the Drowsiness Detection project  
> Evaluates classical machine learning with hand-crafted interpretable features

This experiment extracts 63 features from time, frequency, nonlinear, and EOG-specific domains, training multiple classical ML algorithms for drowsiness classification.

**Key Results**: Ensemble achieves Cohen's Kappa = 0.179, with feature importance insights

## Overview

This experiment implements a **feature engineering + traditional ML approach** for drowsiness detection. Instead of end-to-end deep learning, this approach extracts interpretable features from EOG signals and trains classical machine learning classifiers.

### Key Features

- **Interpretable features**: 70 engineered features across 4 categories
- **Multiple classifiers**: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, Ensemble
- **Same test set**: Subject 07 (fair comparison with Phase 1)
- **Feature importance analysis**: Understand which features matter most
- **Lower computational cost**: No GPU required for inference

### Key Information

- **Window**: 16 seconds (3,200 samples @ 200 Hz) - matches CNN_16s from Phase 1
- **Stride**: 200 samples (1 second) for sliding window
- **Features**: 70 features (26 time-domain, 20 frequency-domain, 10 non-linear, 14 EOG-specific)
- **Models**: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, Gradient Boosting, Ensemble
- **Test Set**: Subject 07 (same as Phase 1 for fair comparison)
- **Train Set**: Subjects 01-06, 08-10 (16 files, no validation set)
- **Best Performance**: Ensemble with Cohen's Kappa = 0.179

## 📁 Directory Structure

```
Phase2_ML/
├── README.md                          # This file
├── feature_engineering/               # Feature extraction modules
│   ├── __init__.py
│   ├── time_domain.py                # Time-domain features (26)
│   ├── frequency_domain.py           # Frequency-domain features (20)
│   ├── nonlinear.py                  # Non-linear features (10)
│   ├── eog_specific.py               # EOG-specific features (14)
│   └── EOG_Feature_Explanations.md   # Feature descriptions
├── features/                          # Extracted feature CSVs
│   ├── train_features_16s.csv        # Training features (~150 MB)
│   └── test_features_16s.csv         # Test features (~20 MB)
├── notebooks/
│   └── ML_Training_Analysis.ipynb    # ML training & analysis
├── scripts/
│   └── ml_training_analysis.py       # Analysis script (.py with cell markers)
├── feature_extraction.py             # Feature extraction script
├── train_ml_colab.py                 # Colab training script
├── requirements_phase2.txt           # Phase 2 dependencies
├── PHASE2_RESULTS_ANALYSIS.md        # Detailed results analysis
└── INSTRUCTIONS.md                   # Additional instructions
```

## 🚀 Quick Start

### Step 1: Feature Extraction (Local - LONG RUNTIME)

Extract features from original EDF files:

```bash
python feature_extraction.py
```

**What it does**:
- Reads EDF files from `../original_data/`
- Applies bandpass filter (0.1-10 Hz)
- Resamples to 200 Hz
- Creates 16s sliding windows (stride=1s)
- Extracts 70 features per window
- Saves CSVs to `features/`

**Runtime**: ~30-60 minutes (depending on CPU)

**Output**:
- `features/train_features_16s.csv` (~150 MB)
- `features/test_features_16s.csv` (~20 MB)

### Step 2: Train ML Models (Google Colab)

1. **Upload** `train_ml_colab.py` to Google Colab
2. **Convert** to notebook (or copy cell-by-cell)
3. **Upload** the two CSV files when prompted:
   - `train_features_16s.csv`
   - `test_features_16s.csv`
4. **Run** all cells

**What it does**:
- Loads feature CSVs
- Standardizes features (zero mean, unit variance)
- Trains 7 classifiers with class balancing
- Evaluates on Subject 07
- Compares with CNN_16s results
- Saves models and plots to Google Drive

**Runtime**: ~10-20 minutes

**Output** (saved to Google Drive):
- Model files: `*.pkl` (Random Forest, SVM, XGBoost, etc.)
- `feature_importance.csv`
- `model_comparison.csv`
- Confusion matrices (PNG)
- Comparison plots (PNG)

### Step 3: Analyze Results

```bash
# In Jupyter or Colab
jupyter notebook notebooks/ML_Training_Analysis.ipynb

# Or as Python script
python scripts/ml_training_analysis.py
```

## 📊 Feature Categories

### Time-Domain (26 features)

**Basic Statistics**:
- mean, std, var, min, max, range, median

**Shape Characteristics**:
- peak_amplitude, valley_amplitude
- rise_time, fall_time
- inflection_points

**Statistical Moments**:
- skewness, kurtosis

**Signal Properties**:
- zero_crossing_rate
- RMS (root mean square)
- crest_factor
- energy

**Hjorth Parameters**:
- activity, mobility, complexity

### Frequency-Domain (20 features)

**Band Powers** (absolute + relative):
- delta (0.5-4 Hz)
- theta (4-8 Hz)
- alpha (8-13 Hz)
- beta (13-30 Hz)

**Spectral Features**:
- spectral_centroid
- spectral_spread
- spectral_entropy
- spectral_edge_frequency

**Band Ratios**:
- theta/alpha
- delta/beta
- alpha/beta
- low_freq/high_freq

### Non-Linear (10 features)

**Entropy Measures**:
- sample_entropy
- approximate_entropy
- permutation_entropy

**Fractal Dimensions**:
- higuchi_fractal_dimension
- petrosian_fractal_dimension
- katz_fractal_dimension
- hurst_exponent

**Complexity Measures**:
- lempel_ziv_complexity
- DFA_alpha (Detrended Fluctuation Analysis)

### EOG-Specific (14 features)

**Blink Detection**:
- blink_rate
- blink_amplitude_mean
- blink_amplitude_std

**Eye Movements**:
- slow_eye_movement_count
- slow_eye_movement_duration
- eye_movement_amplitude_mean
- eye_movement_amplitude_std
- eye_movement_velocity_mean
- eye_movement_velocity_std
- saccade_rate

**Channel Correlation**:
- LOC_ROC_correlation
- LOC_ROC_lag
- LOC_ROC_lag_correlation

**Asymmetry**:
- energy_ratio_LOC_ROC
- amplitude_ratio_LOC_ROC

## 📈 Results Summary

### Model Performance (Test Set: Subject 07)

| Model | Kappa | Precision | Recall | F1-Score | Accuracy |
|-------|-------|-----------|--------|----------|----------|
| **Ensemble** 🏆 | **0.179** | **35.5%** | **49.9%** | **0.415** | 65.4% |
| XGBoost | 0.168 | 33.8% | 52.5% | 0.411 | 63.8% |
| LightGBM | 0.171 | 34.3% | 50.4% | 0.408 | 64.3% |
| Random Forest | 0.116 | 46.6% | 14.2% | 0.217 | 74.9% |
| SVM | 0.103 | 29.4% | 27.3% | 0.283 | 68.5% |
| Gradient Boosting | 0.142 | 31.7% | 45.1% | 0.373 | 62.1% |
| Logistic Regression | -0.006 | 24.4% | 70.9% | 0.363 | 38.6% |

### Comparison with Phase 1 (CNN_16s)

| Metric | Phase 1 (CNN_16s) | Phase 2 (Ensemble) | Difference |
|--------|-------------------|-------------------|------------|
| **Cohen's Kappa** | **0.394** | **0.179** | **-54.6%** |
| **Recall (Drowsy)** | **32.7%** | **49.9%** | **+52.6%** |
| **Precision (Drowsy)** | **54.9%** | **35.5%** | **-35.3%** |
| **F1-Score** | 0.408 | 0.415 | +1.7% |

### Key Insights

**Why ML Underperformed CNN**:
- Engineered features vs. learned representations
- Less data augmentation (stride 200 vs 1)
- Simpler model architecture
- Feature engineering may miss complex patterns

**Advantages of ML Approach**:
- ✅ **Higher recall**: Detects more drowsiness events (49.9% vs 32.7%)
- ✅ **Interpretable features**: Can explain predictions
- ✅ **Feature importance analysis**: Understand what matters
- ✅ **No GPU needed**: Lower computational requirements
- ✅ **Faster training**: 10-20 minutes vs 30-90 minutes
- ✅ **Lower deployment cost**: Smaller models, faster inference

**Best Use Cases**:
- Research applications requiring interpretability
- Resource-constrained environments
- Applications prioritizing recall over precision
- Feature analysis and domain insights

## 🔧 Configuration

### Feature Extraction Settings

```python
WINDOW_SIZE = 16  # seconds
STRIDE = 1        # second (200 samples @ 200 Hz)
SAMPLING_RATE = 200  # Hz
BANDPASS_FILTER = (0.1, 10)  # Hz
```

### ML Training Settings

```python
# Random Forest
n_estimators = 300
class_weight = 'balanced'
max_depth = None

# SVM
kernel = 'rbf'
class_weight = 'balanced'
C = 1.0
gamma = 'scale'

# XGBoost
n_estimators = 300
scale_pos_weight = auto  # Based on class imbalance
max_depth = 6

# LightGBM
n_estimators = 300
class_weight = 'balanced'
max_depth = -1
```

## 📊 Feature Importance Analysis

### Top 10 Most Important Features (Random Forest)

1. **theta_power_relative** (13.2%) - Theta band power
2. **alpha_power_relative** (11.8%) - Alpha band power
3. **spectral_entropy** (9.4%) - Frequency domain entropy
4. **blink_rate** (8.7%) - Eye blink frequency
5. **LOC_ROC_correlation** (7.3%) - Channel correlation
6. **sample_entropy** (6.9%) - Time series complexity
7. **mean** (6.1%) - Signal mean
8. **std** (5.8%) - Signal standard deviation
9. **theta_alpha_ratio** (5.4%) - Theta/Alpha ratio
10. **eye_movement_velocity_mean** (4.9%) - Eye movement speed

### Feature Category Importance

- **Frequency-Domain**: 38% (most important)
- **EOG-Specific**: 28%
- **Time-Domain**: 22%
- **Non-Linear**: 12%

## 🛠️ Troubleshooting

### Common Issues

**1. Feature extraction takes too long**
```python
# Solution: Increase stride (less overlap)
STRIDE = 2  # 2 seconds instead of 1
# This reduces windows by ~50%
```

**2. Memory error during feature extraction**
```python
# Solution: Process files in batches
# Or increase system RAM
```

**3. NaN values in features**
```python
# Solution: Already handled by SimpleImputer in training script
# Uses median imputation
```

**4. Low performance**
```
# Expected - ML typically underperforms CNN
# Focus on interpretability and feature insights
```

**5. Feature CSVs too large for Colab upload**
```bash
# Solution: Compress files before upload
zip -9 features.zip train_features_16s.csv test_features_16s.csv
# Or use Google Drive direct upload
```

## 📖 Detailed Documentation

- **[EOG_Feature_Explanations.md](feature_engineering/EOG_Feature_Explanations.md)**: Detailed feature descriptions
- **[PHASE2_RESULTS_ANALYSIS.md](PHASE2_RESULTS_ANALYSIS.md)**: Comprehensive results analysis
- **[../README.md](../README.md)**: Main project documentation
- **[../Phase1_CNN/README.md](../Phase1_CNN/README.md)**: Phase 1 CNN approach

## 🔬 Technical Details

### Feature Extraction Pipeline

1. **Load EDF file**: Read raw EOG signals
2. **Bandpass filter**: 0.1-10 Hz (remove DC offset and high-frequency noise)
3. **Resample**: 128 Hz → 200 Hz (match Phase 1)
4. **Sliding window**: 16-second windows with 1-second stride
5. **Extract features**: Compute all 70 features per window
6. **Label**: Assign Awake/Drowsy label based on annotations
7. **Save**: Write to CSV with metadata

### ML Training Pipeline

1. **Load features**: Read CSV files
2. **Handle NaN**: Median imputation
3. **Standardize**: Zero mean, unit variance
4. **Train models**: With class balancing
5. **Evaluate**: On Subject 07 test set
6. **Feature importance**: Analyze top features
7. **Compare**: With Phase 1 CNN results
8. **Save**: Models and results to Drive

## 🎯 Future Improvements

1. **Feature Selection**: Use feature importance to reduce dimensionality
2. **Hyperparameter Tuning**: Grid search for optimal parameters
3. **More Models**: Try CatBoost, Neural Networks
4. **Different Windows**: Experiment with 8s or 32s windows
5. **Ensemble Methods**: Stacking, blending
6. **SMOTE**: Oversample minority class
7. **Feature Engineering**: Create interaction features
8. **SHAP Values**: Advanced interpretability analysis

## 📄 File Descriptions

### Feature Engineering Modules

- `feature_engineering/time_domain.py`: Time-domain feature extraction
- `feature_engineering/frequency_domain.py`: Frequency-domain feature extraction
- `feature_engineering/nonlinear.py`: Non-linear feature extraction
- `feature_engineering/eog_specific.py`: EOG-specific feature extraction
- `feature_engineering/__init__.py`: Module initialization

### Scripts

- `feature_extraction.py`: Main feature extraction script
- `train_ml_colab.py`: Colab training script
- `scripts/ml_training_analysis.py`: Analysis script with cell markers

### Notebooks

- `notebooks/ML_Training_Analysis.ipynb`: Complete ML training and analysis

### Documentation

- `README.md`: This file
- `PHASE2_RESULTS_ANALYSIS.md`: Detailed results analysis
- `INSTRUCTIONS.md`: Additional instructions
- `EOG_Feature_Explanations.md`: Feature descriptions

## 📞 Support

For issues or questions:
- Check troubleshooting section above
- Review main project README
- Check feature explanations
- Review results analysis document
- Open an issue on GitHub

---

**Phase 2 Focus**: Feature engineering and traditional machine learning for drowsiness detection, demonstrating the trade-offs between interpretability, computational efficiency, and performance compared to end-to-end deep learning.
