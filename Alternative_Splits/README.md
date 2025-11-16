# Alternative Data Splitting Experiments

This directory contains experiments with different data splitting strategies to evaluate model generalization beyond the original Phase 1 cross-subject split.

**Key Innovation**: Tests multiple evaluation strategies (temporal, subject-independent, and hybrid) to comprehensively assess model performance.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Experiments](#experiments)
   - [Experiment 1: Within-Subject Split](#experiment-1-within-subject-split)
   - [Experiment 2: Leave-One-Subject-Out CV](#experiment-2-leave-one-subject-out-cv)
   - [Experiment 3: Split 1.5 Recordings](#experiment-3-split-15-recordings-best-results)
4. [CNN Training](#cnn-training)
5. [ML Training (Optional)](#ml-training-optional)
6. [Results Summary](#results-summary)
7. [Quick Start (עברית)](#quick-start-עברית)

---

## 🎯 Overview

### Purpose

Test alternative evaluation strategies to better understand:
1. **Temporal Generalization**: Can model generalize across time (within subjects)?
2. **Subject-Independent Generalization**: Can model generalize to new subjects?
3. **Optimal Training Size**: Does more training data (1.5 recordings) improve performance?

### Why Alternative Splits?

Original Phase 1 used **cross-subject split** (Train: Subjects 1-6, 8-10 | Test: Subject 7):
- ✅ Tests subject-independent generalization
- ❌ Limited to one subject in test set
- ❌ Only 50% of data used for training

Alternative splits explore:
- More training data (Split 1.5)
- Temporal generalization (Within-Subject)
- Robust cross-subject validation (10-fold CV)

---

## 📁 Directory Structure

```
Alternative_Splits/
├── README.md                                    # This file
├── RESULTS_SUMMARY.md                           # Comprehensive results analysis
│
├── scripts/                                     # Split generation utilities
│   ├── create_within_subject_split.py          # Generate within-subject split
│   └── create_cv_folds.py                      # Generate 10-fold CV splits
│
├── within_subject/                              # Experiment 1 data
│   ├── file_sets.mat                           # Split definition
│   └── train_colab.py                          # Training script
│
├── cross_validation/                            # Experiment 2 data
│   ├── fold_01.mat ... fold_10.mat            # 10 fold definitions
│   └── train_cv_colab.py                       # CV training script
│
├── Alternative_Splits_Training_Colab_v2.py     # CNN training (Exp 1 & 2)
├── Split_1_5_Training_Colab.py                 # CNN training (Exp 3)
├── Alternative_Splits_Feature_Extraction.py    # ML feature extraction (optional)
├── Display_Detailed_Results.py                 # Results visualization
│
└── ml_features/                                 # ML training features (gitignored)
    ├── split_1_5_train_features_16s.csv
    ├── split_1_5_test_features_16s.csv
    ├── within_subject_train_features_16s.csv
    └── within_subject_test_features_16s.csv
```

---

## 🔬 Experiments

---

### Experiment 1: Within-Subject Split

#### Concept
- **Train**: All _1 recordings (10 files)
- **Test**: All _2 recordings (10 files)
- **Type**: Temporal generalization within subjects

#### Advantages
- ✅ Fast (single run: ~1 hour)
- ✅ Balanced data split
- ✅ Tests temporal robustness
- ✅ All subjects represented

#### Considerations
- ⚠️ Same subjects in train/test (not subject-independent)
- ⚠️ Temporal differences may affect performance

#### Results (CNN_16s)
- **Cohen's Kappa**: 0.3227
- **Drowsy Recall**: 28.73%
- **Drowsy Precision**: 38.79%

---

### Experiment 2: Leave-One-Subject-Out CV

#### Concept
- **10 Folds**: Each subject tested once
- **Fold N**: Test = Subject N, Train = All others
- **Type**: Subject-independent generalization

#### Advantages
- ✅ Gold standard evaluation
- ✅ Robust statistics (mean ± std)
- ✅ True generalization to new subjects
- ✅ No data leakage

#### Considerations
- ⏱️ Time-consuming (10 runs: 5-15 hours)
- 📊 Variability across folds

#### Results (CNN_16s - Average)
- **Cohen's Kappa**: Variable across subjects
- **Purpose**: Comprehensive cross-subject validation

---

### Experiment 3: Split 1.5 Recordings (BEST RESULTS!)

#### Concept
- **Train**: All _1 (full) + All _2 (first 50%)
- **Test**: All _2 (last 50%)
- **Type**: Hybrid - temporal generalization with more data

#### Advantages
- ✅ **50% more training data** than Within-Subject
- ✅ Still tests temporal generalization
- ✅ Best performance achieved
- ✅ Single run (~1 hour)

#### Why Best?
More training data leads to better generalization while still testing on unseen temporal data.

#### Results (CNN_16s) 🏆
- **Cohen's Kappa**: **0.4328** (BEST!)
- **Drowsy Recall**: 34.73%
- **Drowsy Precision**: 41.84%
- **F1-Score**: 0.3796

---

## 🚀 CNN Training

### Option 1: Unified Script (Experiments 1 & 2)

**File**: `Alternative_Splits_Training_Colab_v2.py`

Trains both Within-Subject and Cross-Validation experiments.

```python
# In the script, set:
EXPERIMENT = 'within_subject'  # or 'cross_validation' or 'both'

# Upload to Google Colab and run
%run Alternative_Splits_Training_Colab_v2.py
```

**Configuration**:
- Model: CNN_16s
- Epochs: 5
- Batch size: 800
- Window: 16 seconds
- Results: Saved to Google Drive

### Option 2: Split 1.5 Script (Experiment 3) - RECOMMENDED

**File**: `Split_1_5_Training_Colab.py`

Optimized for best performance.

```python
# Upload to Google Colab and run
%run Split_1_5_Training_Colab.py
```

**Runtime**: ~1 hour  
**Best Results**: Kappa 0.43

---

## 🤖 ML Training (Optional)

For comparison with traditional ML approaches.

### Step 1: Extract Features (Local)

```bash
cd Alternative_Splits
python Alternative_Splits_Feature_Extraction.py
```

**Runtime**: ~15-20 minutes  
**Output**: 4 CSV files in `ml_features/`

**Features Extracted** (63 total):
- Time-domain: 26 features
- Frequency-domain: 16 features
- Non-linear: 7 features
- EOG-specific: 12 features

### Step 2: Train ML Models (Colab)

Upload your ML training script (adapted from Phase2_ML) to Colab.

**Models Tested**:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Ensemble (weighted combination)

**Best ML Model** (Split 1.5): LightGBM
- Kappa: 0.203 (still lower than CNN's 0.43)
- Recall: 20.3%
- Precision: 32.6%

---

## 📊 Results Summary

### Complete Comparison - Split 1.5 Recordings

| Model | Cohen's Kappa | Drowsy Recall | Drowsy Precision | F1-Score |
|-------|---------------|---------------|------------------|----------|
| **CNN_16s** 🥇 | **0.4328** | **34.73%** | **41.84%** | **0.3796** |
| LightGBM | 0.2033 | 20.27% | 32.55% | 0.2499 |
| XGBoost | 0.1889 | 18.14% | 32.58% | 0.2330 |
| Random Forest | 0.1444 | 12.42% | 32.82% | 0.1802 |
| SVM | 0.0222 | 64.37% | 10.08% | 0.1743 |

### Key Findings

1. **CNN significantly outperforms ML models**
   - 113% better Kappa than best ML (LightGBM)
   - 71% better recall
   - 28% better precision

2. **Split 1.5 achieves best results**
   - 34% better Kappa than Within-Subject
   - More training data = better performance

3. **LightGBM best among ML models**
   - Faster training (30s vs 1h for CNN)
   - Reasonable performance for rapid prototyping

### Detailed Results

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for:
- Per-class metrics for each model
- Confusion matrices
- Cross-experiment comparisons
- Statistical analysis
- Recommendations

---

## 🔍 Visualizing Results

Use the included results viewer:

```python
# In Colab or local Python
python Display_Detailed_Results.py

# Configure which experiment to display:
EXPERIMENT = 'split_1_5'  # or 'within_subject' or 'both'
```

**Displays**:
- Per-class precision, recall, F1
- Detailed confusion matrices
- Sensitivity & specificity
- False positive/negative rates

---

## ⚡ Quick Start (עברית)

### הרצה מהירה - ניסוי Split 1.5 (מומלץ)

#### שלב 1: העלאה ל-Colab

1. פתח Google Colab
2. העלה את `Split_1_5_Training_Colab.py`
3. וודא שהמידע מ-Drive זמין

#### שלב 2: הרצת אימון

```python
# הרץ את הסקריפט
%run Split_1_5_Training_Colab.py
```

**זמן ריצה**: בערך שעה

#### שלב 3: תוצאות

התוצאות נשמרות אוטומטית ל-Google Drive:
```
/content/drive/MyDrive/drowsiness-detection_-eog-based/
└── alternative_splits_results/
    └── split_1_5/
        ├── test_results.mat
        └── best_model.h5
```

#### התוצאות המצופות

```
Cohen's Kappa:  0.43   (מצוין!)
Drowsy Recall:  34.7%  (זיהוי drowsy)
Drowsy Precision: 41.8% (דיוק drowsy)
```

---

### אימון ML (אופציונלי)

#### שלב 1: חילוץ Features (מקומי)

```bash
cd Alternative_Splits
python Alternative_Splits_Feature_Extraction.py
```

**זמן**: 15-20 דקות  
**פלט**: 4 קבצי CSV

#### שלב 2: אימון (Colab)

1. העלה את 4 קבצי ה-CSV ל-Colab
2. הרץ את סקריפט האימון שלך (מבוסס Phase2)
3. תוצאות: LightGBM הכי טוב (Kappa: 0.20)

---

### הצגת תוצאות

```python
python Display_Detailed_Results.py --experiment split_1_5 --local
```

---

## 🛠️ Technical Details

### Data Splits

**Within-Subject**:
- Files: All `*_1.mat` → train, All `*_2.mat` → test
- Generated by: `scripts/create_within_subject_split.py`

**Split 1.5**:
- Files: All `*_1.mat` (full) + `*_2.mat` (first 50%) → train
- Files: `*_2.mat` (last 50%) → test
- Split point: `len(signal) // 2`

**Cross-Validation**:
- 10 folds, each excluding one subject
- Generated by: `scripts/create_cv_folds.py`

### Model Configuration

**CNN_16s**:
- Window: 16 seconds (3200 samples @ 200 Hz)
- Channels: 2 (E1, E2)
- Architecture: Conv layers → BatchNorm → MaxPool → Dense
- Optimizer: Nadam
- Loss: Categorical cross-entropy
- Class weights: Balanced

**ML Models**:
- Features: 63 per window
- Window: 16 seconds, Stride: 8 seconds (50% overlap)
- Class weights: Balanced for all models

---

## 📚 Related Files

- **Phase1_CNN**: Original CNN implementation
- **Phase2_ML**: ML baseline (Logistic Regression, RF, XGBoost, LightGBM)
- **RESULTS_SUMMARY.md**: Detailed results with all comparisons
- **Main README.md**: Project overview (root directory)

---

## 🎯 Recommendations

### For Research Papers
- **Report**: Split 1.5 results (Kappa: 0.43)
- **Comparison**: Mention ML baseline (LightGBM: 0.20)
- **Cite**: Alternative splits for robustness

### For Practical Deployment
- **Use**: Split 1.5 trained model (best performance)
- **Consider**: LightGBM for rapid prototyping (10x faster training)
- **Warning**: Current recall (35%) insufficient for safety-critical use alone

### For Future Work
- **Improve**: Drowsy detection recall (target >70%)
- **Explore**: Ensemble of CNN + ML
- **Test**: Multi-modal approaches (EOG + other sensors)

---

## ✅ Success Criteria

For each experiment:
- ✅ Model trains successfully
- ✅ Kappa > 0.20 (fair agreement)
- ✅ Recall > 20% (drowsy detection)
- ✅ Results saved and reproducible

**Best Model**: Split 1.5 CNN_16s
- ✅ Kappa: 0.43 (moderate agreement)
- ✅ Recall: 34.7% (best drowsy detection)
- ✅ Reproducible results

---

## 📞 Troubleshooting

### Issue: "File not found" errors
**Solution**: Ensure MAT files exist in `Phase1_CNN/data/processed/files/`

### Issue: Training very slow
**Solution**: Verify GPU is enabled in Colab (Runtime → Change runtime type)

### Issue: Out of memory
**Solution**: Reduce batch size in script (default: 800 → try 400)

### Issue: Results differ from reported
**Explanation**: Random seed variations are normal (±0.02 Kappa is acceptable)

---

## 📖 Citation

If using these experiments in your research, please cite:
- Original dataset: Dryad repository
- Methodology: Alternative splitting strategies
- Results: Split 1.5 achieves best performance (Kappa: 0.43)

---

## 📅 Version History

- **v2.0** (November 2024): Added Split 1.5 + ML training pipeline
- **v1.0** (Initial): Within-Subject + Cross-Validation experiments

---

**Last Updated**: November 2024  
**Status**: Complete and validated  
**Best Results**: Split 1.5 with CNN_16s (Kappa: 0.4328)

---

For detailed implementation, see individual script files.  
For comprehensive results analysis, see [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md).
