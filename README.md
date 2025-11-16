# Drowsiness Detection from EOG Signals: Three Complementary Approaches

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)


## 🎯 Project Overview

This project evaluates **three complementary approaches** for automatic drowsiness detection from EOG (Electrooculography) signals using the public Dryad dataset.

### Three Experiments

#### Experiment 1: CNN with Cross-Subject Evaluation
**Directory**: `Phase1_CNN/`

- **Approach**: Deep learning with convolutional neural networks
- **Validation**: Subject-independent (train on 9 subjects, test on 1 unseen)
- **Models**: 6 CNN variants (2s, 4s, 8s, 16s, 32s windows + CNN-LSTM)
- **Key Strength**: Tests true generalization to completely new individuals
- **Best Result**: CNN_16s, Cohen's Kappa = 0.394

#### Experiment 2: Traditional ML with Feature Engineering
**Directory**: `Phase2_ML/`

- **Approach**: Hand-crafted features with classical machine learning
- **Features**: 63 features (time, frequency, nonlinear, EOG-specific domains)
- **Models**: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM, Ensemble
- **Key Strength**: Interpretability and computational efficiency
- **Best Result**: Ensemble, Cohen's Kappa = 0.179

#### Experiment 3: CNN with Multiple Splitting Strategies
**Directory**: `Alternative_Splits/`

- **Approach**: Same CNN architecture with optimized data splitting
- **Strategies**: Within-Subject, 10-fold Cross-Validation, Split 1.5 Recordings
- **Key Strength**: Maximizes training data while testing temporal generalization
- **Best Result**: CNN_16s Split 1.5, Cohen's Kappa = 0.433 🏆 **PROJECT BEST**

### Key Findings

- **Best Overall Performance**: Experiment 3 (Split 1.5) — Kappa = 0.433
- **CNN significantly outperforms ML**: ~120% better Kappa
- **Data splitting matters**: Optimized strategy improves CNN by 10%
- **Practical deployment**: EOG-only setup achieves moderate agreement (Kappa 0.4-0.5)

### Recommended Reading Order

1. Start with **Experiment 3** (`Alternative_Splits/`) for best results
2. Review **Experiment 1** (`Phase1_CNN/`) for baseline CNN approach
3. Explore **Experiment 2** (`Phase2_ML/`) for interpretable features

See `EXPERIMENTS_OVERVIEW.md` for detailed comparison.

### Dataset

All experiments use the [Dryad drowsiness dataset](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c): 10 subjects, 20 recording sessions, 816 drowsiness events, 2 EOG channels only.

## 📊 Dataset Description

**Source**: [Dryad Drowsiness Dataset](https://datadryad.org/dataset/doi:10.5061/dryad.5tb2rbp9c)

### Dataset Characteristics

- **Subjects**: 10 participants (20 recording sessions total)
- **Drowsiness Events**: 816 annotated drowsiness episodes
- **Channels**: 2 EOG channels (LOC-Ref, ROC-Ref)
- **Original Sampling Rate**: 128 Hz
- **Resampled to**: 200 Hz (for CNN compatibility)
- **Original Format**: EDF files with separate annotation EDF files
- **Class Imbalance**: ~96.5% Awake vs. ~3.5% Drowsy (severe imbalance)

### Channel Selection Rationale

**Original Dryad Dataset** is a comprehensive polysomnography recording including:
- **EEG channels** (2): Fpz-Cz, Pz-Oz (brain activity)
- **EOG channels** (2): LOC-Ref, ROC-Ref (eye movements) ← **Used in this project**
- **EMG channels** (1): Chin EMG (muscle activity)
- **ECG** (1): Single-lead electrocardiogram (heart activity)
- **Respiration** (2): Thoracic and abdominal effort

**This project exclusively uses the 2 EOG channels** for the following reasons:

1. **Minimal Invasiveness**: EOG electrodes require only 4 simple skin-surface sensors (vs. 20+ for full EEG cap), making it practical for real-world deployment

2. **Direct Physiological Indicator**: Eye movements and blink patterns are direct, observable indicators of drowsiness:
   - Slow rolling eye movements (SREMs)
   - Prolonged eyelid closures (PERCLOS)
   - Reduced blink frequency

3. **Practical Deployment**: EOG-based detection is feasible in real-world applications:
   - Automotive (driver drowsiness detection)
   - Aviation (pilot monitoring)
   - Occupational safety (operators, security guards)

4. **Computational Efficiency**: 2 channels vs. 20+ channels significantly reduces:
   - Data transmission requirements
   - Storage needs
   - Processing computational cost
   - Real-time inference latency

5. **Research Question**: Evaluating whether a minimal sensor setup (EOG-only) can achieve acceptable drowsiness detection accuracy

All other channels (EEG, EMG, ECG, respiration) were excluded during preprocessing. The raw EDF files contain all modalities, but this project's preprocessing pipeline extracts and processes only the EOG channels.

**Trade-off**: While multi-modal approaches might achieve higher accuracy, this project prioritizes practicality and deployability over maximal performance.

### Labeling Methodology

Drowsiness events are labeled using a **2-second context window** approach:
- Each annotated drowsiness event is extended by ±1 second
- This creates a 2-second buffer around each event
- Samples within this window are labeled as "Drowsy" (1)
- All other samples are labeled as "Awake" (0)

This labeling strategy captures the transition periods into and out of drowsiness states, providing more training data while maintaining clinical relevance.

### Data Split Strategy

**Subject-based splitting** to prevent data leakage:
- **Training**: Subjects 01-06 (12 files, ~35M samples)
- **Validation**: Subjects 08-10 (6 files, ~9M samples)  
- **Test**: Subject 07 (2 files, 100,600 drowsy samples)

Subject 07 was chosen as the test set because it contains the highest number of drowsiness events (100,600 samples), providing a robust evaluation dataset.

## 🏗️ Project Structure

```
drowsiness-detection_-eog-based/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
│
├── Phase1_CNN/                       # Phase 1: Deep Learning Approach
│   ├── README.md                     # Phase 1 documentation
│   ├── data/                         # Processed .mat files
│   │   ├── processed/files/          # 20 .mat files (240 MB)
│   │   ├── file_sets.mat            # Train/val/test split
│   │   └── create_file_sets.py      # Split configuration
│   ├── src/                          # Model source code
│   │   ├── loadData.py              # Data loading utilities
│   │   ├── utils.py                 # Helper functions
│   │   └── models/                  # Model architectures
│   │       ├── cnn/                 # CNN variants (2s, 4s, 8s, 16s, 32s)
│   │       └── cnn_lstm/            # CNN-LSTM hybrid
│   ├── preprocessing/                # Data preprocessing
│   │   ├── edf_to_mat_eog_only.py  # EDF → MAT conversion
│   │   ├── eda.py                   # Exploratory data analysis
│   │   └── README.md
│   ├── notebooks/
│   │   ├── CNN_Training_Colab.ipynb        # Training notebook
│   │   └── CNN_Results_Analysis.ipynb      # Results analysis
│   └── scripts/
│       ├── cnn_training_colab.py           # Training script (.py with cell markers)
│       ├── cnn_results_analysis.py         # Analysis script (.py with cell markers)
│       ├── data_exploration.py             # EDA script (.py with cell markers)
│       └── prepare_for_colab.py            # Package for Colab upload
│
├── Phase2_ML/                        # Phase 2: Feature Engineering Approach
│   ├── README.md                     # Phase 2 documentation
│   ├── feature_engineering/          # Feature extraction modules
│   │   ├── time_domain.py           # Time-domain features (26)
│   │   ├── frequency_domain.py      # Frequency-domain features (20)
│   │   ├── nonlinear.py             # Non-linear features (10)
│   │   ├── eog_specific.py          # EOG-specific features (14)
│   │   └── EOG_Feature_Explanations.md
│   ├── features/                     # Extracted feature CSVs
│   │   ├── train_features_16s.csv   # Training features (~150 MB)
│   │   └── test_features_16s.csv    # Test features (~20 MB)
│   ├── notebooks/
│   │   └── ML_Training_Analysis.ipynb      # ML training & analysis
│   ├── scripts/
│   │   └── ml_training_analysis.py         # Analysis script (.py with cell markers)
│   ├── feature_extraction.py        # Feature extraction script
│   ├── train_ml_colab.py           # Colab training script
│   └── requirements_phase2.txt      # Phase 2 dependencies
│
├── Alternative_Splits/              # Alternative Data Splitting Experiments
│   ├── README.md                     # Alternative splits documentation
│   ├── RESULTS_SUMMARY.md           # Comprehensive results analysis
│   ├── Alternative_Splits_Training_Colab_v2.py  # CNN training (Exp 1 & 2)
│   ├── Split_1_5_Training_Colab.py  # CNN training (Split 1.5 - BEST)
│   ├── Alternative_Splits_Feature_Extraction.py # ML feature extraction
│   ├── Display_Detailed_Results.py  # Results visualization
│   ├── scripts/                      # Split generation utilities
│   │   ├── create_within_subject_split.py
│   │   └── create_cv_folds.py
│   ├── within_subject/               # Experiment 1: Temporal generalization
│   │   └── file_sets.mat
│   └── cross_validation/             # Experiment 2: 10-fold LOSO CV
│       ├── fold_01.mat ... fold_10.mat
│       └── train_cv_colab.py
│
├── original_data/                    # Raw EDF files (optional)
└── archive/                          # Archived/old files
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x / Keras 3.x
NumPy, SciPy, scikit-learn
MNE, pyedflib (for preprocessing)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drowsiness-detection-eog.git
cd drowsiness-detection-eog

# Install dependencies
pip install -r requirements.txt
```

## 📖 Complete Workflow

### Step 1: Data Exploration (Optional but Recommended)

Understand the dataset, labeling methodology, and class imbalance:

```bash
# Run EDA script
python Phase1_CNN/scripts/data_exploration.py

# Or use the preprocessing EDA
python Phase1_CNN/preprocessing/eda.py
```

**What you'll see**:
- Dataset overview (10 subjects, 20 sessions)
- Drowsiness annotation distribution across subjects
- Class imbalance visualization (96.5% vs 3.5%)
- Signal characteristics and quality assessment
- Train/val/test split rationale

### Step 2: Data Preprocessing (If Starting from Raw EDF Files)

Convert raw EDF files to processed MAT format:

```bash
cd Phase1_CNN/preprocessing
python edf_to_mat_eog_only.py
```

**What it does**:
- Reads EDF files and drowsiness annotations
- Extracts 2 EOG channels (LOC-Ref, ROC-Ref)
- Resamples from 128 Hz → 200 Hz with anti-aliasing
- Creates binary labels with 2-second context windows
- Converts to `.mat` format for model compatibility

**Output**: `Phase1_CNN/data/processed/files/*.mat` (20 files, ~240 MB total)

**Runtime**: ~10-15 minutes

### Step 3: Phase 1 - CNN Training

Train deep learning models on Google Colab (recommended) or locally.

#### Option A: Google Colab (Recommended)

1. **Package files for Colab**:
```bash
cd Phase1_CNN/scripts
python prepare_for_colab.py --auto
```

2. **Upload to Colab**:
   - Upload `microsleep_colab_package.zip` to Colab
   - Upload `Phase1_CNN/notebooks/CNN_Training_Colab.ipynb`

3. **Run training**:
   - Select model (CNN_2s, CNN_4s, CNN_8s, CNN_16s, CNN_32s, CNN_LSTM)
   - Set hyperparameters (epochs, batch size, stride)
   - Train on free GPU (T4)
   - Models auto-save to Google Drive

**Recommended settings**:
- **Model**: CNN_16s (best performance)
- **Epochs**: 3-6
- **Batch Size**: 800 (for T4 GPU)
- **Stride**: 1 (maximum data augmentation)

**Runtime**: ~30-90 minutes per model

#### Option B: Local Training

```bash
cd Phase1_CNN/src/models/cnn/CNN_16s
python train.py
```

**Note**: Requires GPU. CPU training is very slow (~hours per epoch).

### Step 4: Phase 1 - Results Analysis

Analyze CNN model performance:

```bash
# In Jupyter or Google Colab
jupyter notebook Phase1_CNN/notebooks/CNN_Results_Analysis.ipynb

# Or run as Python script
python Phase1_CNN/scripts/cnn_results_analysis.py
```

**What you'll see**:
- Performance comparison across all 6 models
- Cohen's Kappa analysis (primary metric for imbalanced data)
- Confusion matrices
- Training curves (loss, accuracy over epochs)
- Window size impact analysis
- Model recommendations

**Key Results** (Test Set: Subject 07):

| Model | Kappa | Precision | Recall | Accuracy |
|-------|-------|-----------|--------|----------|
| **CNN_16s** 🏆 | **0.394** | **54.9%** | **32.7%** | 96.8% |
| CNN_8s | 0.289 | 58.9% | 20.2% | 96.8% |
| CNN_32s | 0.286 | 53.1% | 32.2% | 96.7% |
| CNN_4s | 0.046 | 19.7% | 3.2% | 96.2% |
| CNN_2s | 0.056 | 9.5% | 7.7% | 94.3% |
| CNN_LSTM | 0.008 | 4.2% | 4.6% | 93.1% |

### Step 5: Phase 2 - Feature Extraction

Extract engineered features from EOG signals:

```bash
cd Phase2_ML
python feature_extraction.py
```

**What it does**:
- Applies 16-second sliding windows (stride=1 second)
- Extracts ~70 features per window:
  - 26 time-domain features
  - 20 frequency-domain features
  - 10 non-linear features
  - 14 EOG-specific features
- Saves features to CSV files

**Output**:
- `Phase2_ML/features/train_features_16s.csv` (~150 MB)
- `Phase2_ML/features/test_features_16s.csv` (~20 MB)

**Runtime**: ~30-60 minutes

### Step 6: Phase 2 - ML Training & Analysis

Train traditional ML classifiers:

```bash
# In Google Colab (recommended)
# Upload Phase2_ML/notebooks/ML_Training_Analysis.ipynb
# Upload the two CSV files from Phase2_ML/features/

# Or run training script
python Phase2_ML/train_ml_colab.py
```

**What it does**:
- Trains multiple classifiers:
  - Logistic Regression (baseline)
  - Random Forest (primary)
  - SVM with RBF kernel
  - XGBoost
  - LightGBM
  - Gradient Boosting
  - Ensemble (voting)
- Evaluates on Subject 07 (same test set as Phase 1)
- Analyzes feature importance
- Compares with CNN results

**Runtime**: ~10-20 minutes

**Key Results** (Test Set: Subject 07):

| Model | Kappa | Precision | Recall | F1-Score |
|-------|-------|-----------|--------|----------|
| **Ensemble** 🏆 | **0.179** | **35.5%** | **49.9%** | 0.415 |
| Random Forest | 0.116 | 46.6% | 14.2% | 0.217 |
| XGBoost | 0.168 | 33.8% | 52.5% | 0.411 |
| LightGBM | 0.171 | 34.3% | 50.4% | 0.408 |
| SVM | 0.103 | 29.4% | 27.3% | 0.283 |
| Logistic Regression | -0.006 | 24.4% | 70.9% | 0.363 |

### Step 7: Alternative Splits - Extended Evaluation

Test different data splitting strategies for comprehensive generalization assessment:

```bash
# Split 1.5 (BEST RESULTS - RECOMMENDED) 🏆
cd Alternative_Splits/training_scripts
# Upload Split_1_5_Training_Colab.py to Google Colab and run
```

**What it does**:
- Tests 3 different splitting strategies:
  1. **Split 1.5 Recordings**: Train on _1 + first 50% of _2, test on last 50% of _2
  2. **Within-Subject Split**: Train on _1, test on _2 recordings  
  3. **10-Fold Cross-Validation**: Leave-one-subject-out (LOSO)

**Scripts**:
- `training_scripts/Split_1_5_Training_Colab.py` — Best performance
- `training_scripts/Alternative_Splits_Training_Colab_v2.py` — Within-Subject + CV
- `analysis/Display_Detailed_Results.py` — View detailed metrics

**Key Results** (CNN_16s):

| Strategy | Cohen's Kappa | Drowsy Recall | Drowsy Precision |
|----------|---------------|---------------|------------------|
| **Split 1.5** 🥇 | **0.433** | **34.7%** | **41.8%** |
| Within-Subject | 0.323 | 28.7% | 38.8% |
| Cross-Subject (Original) | 0.394 | 32.7% | 54.9% |

**Why Split 1.5 is Best**: 50% more training data + temporal generalization testing

See [Alternative_Splits/README.md](Alternative_Splits/README.md) for detailed instructions.

## 📈 Results Comparison: All Approaches

### Performance Summary (Best Models from Each Approach)

| Approach | Method | Cohen's Kappa | Recall | Precision | F1-Score |
|----------|--------|---------------|--------|-----------|----------|
| **Alternative Splits** 🥇 | CNN_16s (Split 1.5) | **0.433** | 34.7% | 41.8% | **0.380** |
| **Phase 1 (Original)** | CNN_16s (Cross-Subject) | 0.394 | 32.7% | 54.9% | 0.408 |
| **Alternative Splits** | CNN_16s (Within-Subject) | 0.323 | 28.7% | 38.8% | 0.330 |
| **Phase 2** | Ensemble ML | 0.179 | **49.9%** | 35.5% | 0.415 |

### Interpretation

**Phase 1 (CNN) Advantages**:
- ✅ Higher overall agreement (Kappa)
- ✅ Higher precision (fewer false alarms)
- ✅ Better for applications requiring high confidence
- ✅ End-to-end learning captures complex patterns

**Phase 2 (ML) Advantages**:
- ✅ Higher recall (detects more drowsiness events)
- ✅ Interpretable features (feature importance analysis)
- ✅ No GPU required for inference
- ✅ Faster training
- ✅ Lower computational cost

### Use Case Recommendations

- **Safety-critical applications** (e.g., driver monitoring): Use **Phase 1 CNN** for higher precision
- **Research & analysis**: Use **Phase 2 ML** for interpretability and feature insights
- **Resource-constrained deployment**: Use **Phase 2 ML** for lower computational requirements
- **Maximum detection**: Use **Phase 2 ML** for higher recall

## 🔧 Technical Details

### Phase 1: CNN Architecture

Deep convolutional neural network with:
- **Input**: EOG signals (2 channels × window_size samples)
- **Architecture**:
  - 12 Conv2D layers (32→256 filters)
  - Batch Normalization + ReLU activation
  - MaxPooling2D (stride 2)
  - Gaussian Noise (σ=0.1) for regularization
  - 2 Dense layers (256→2)
  - Dropout (0.25)
- **Output**: Binary classification (Awake/Drowsy)

**Window sizes**:
- CNN_2s: 400 samples (2 sec × 200 Hz)
- CNN_4s: 800 samples (4 sec)
- CNN_8s: 1,600 samples (8 sec)
- **CNN_16s**: 3,200 samples (16 sec) ⭐ **Best**
- CNN_32s: 6,400 samples (32 sec)

### Phase 2: Feature Engineering

**70 features** across 4 categories:

1. **Time-Domain (26 features)**:
   - Basic stats: mean, std, var, min, max, range, median
   - Shape: peak/valley amplitude, rise/fall time
   - Statistical: skewness, kurtosis, RMS, crest factor
   - Advanced: zero-crossing rate, energy
   - Hjorth: activity, mobility, complexity

2. **Frequency-Domain (20 features)**:
   - Band powers: delta, theta, alpha, beta (absolute + relative)
   - Spectral: centroid, spread, entropy, edge frequency
   - Ratios: theta/alpha, delta/beta, alpha/beta

3. **Non-Linear (10 features)**:
   - Entropy: sample, approximate, permutation
   - Fractal: Higuchi, Petrosian, Katz, Hurst exponent
   - Complexity: Lempel-Ziv, DFA alpha

4. **EOG-Specific (14 features)**:
   - Blinks: rate, amplitude (mean, std)
   - Slow eye movements: count, duration
   - LOC-ROC: correlation, lag, lag correlation
   - Movements: amplitude, velocity, saccade rate
   - Asymmetry: energy ratio, amplitude ratio

### Class Imbalance Handling

Both phases address the severe class imbalance (96.5% vs 3.5%):

**Phase 1 (CNN)**:
- Class weights: Drowsy samples weighted 30-40x more
- Sample weights in data generator
- Cohen's Kappa as primary metric

**Phase 2 (ML)**:
- `class_weight='balanced'` in all classifiers
- Stratified sampling
- Cohen's Kappa as primary metric

## 📚 Documentation

- **[Phase1_CNN/README.md](Phase1_CNN/README.md)**: Detailed Phase 1 documentation
- **[Phase2_ML/README.md](Phase2_ML/README.md)**: Detailed Phase 2 documentation
- **[Alternative_Splits/README.md](Alternative_Splits/README.md)**: Alternative splitting experiments (Within-Subject, CV, Split 1.5)
- **[Alternative_Splits/RESULTS_SUMMARY.md](Alternative_Splits/RESULTS_SUMMARY.md)**: Comprehensive results analysis & comparison
- **[Phase1_CNN/preprocessing/README.md](Phase1_CNN/preprocessing/README.md)**: Preprocessing guide
- **[Phase2_ML/feature_engineering/EOG_Feature_Explanations.md](Phase2_ML/feature_engineering/EOG_Feature_Explanations.md)**: Feature descriptions

## 🛠️ Troubleshooting

### Common Issues

**1. GPU not found in Colab**
```python
# Solution: Runtime → Change runtime type → GPU (T4)
```

**2. Out of memory during training**
```python
# Solution: Reduce batch size
BATCH_SIZE = 400  # Instead of 800
```

**3. "File not found" errors**
```bash
# Solution: Verify paths are correct
# Check that data files exist in expected locations
```

**4. Class imbalance warnings**
```
# Normal behavior - class weights handle this automatically
# Check training output for class weight values
```

**5. Low performance on validation set**
```
# Expected - validation subjects may differ from training distribution
# Focus on test set (Subject 07) as primary evaluation
```

## 📄 Attribution, Citations & Original Work

### Architectural Foundation

This project adapts the CNN architecture from:

**Malafeev, A., Hertig-Godeschalk, A., Schreier, D. R., Skorucak, J., Mathis, J., & Achermann, P. (2021).** Automatic Detection of Microsleep Episodes with Deep Learning. *Frontiers in Neuroscience*, 15, 564098. https://doi.org/10.3389/fnins.2021.564098

```bibtex
@article{malafeev2021automatic,
  title={Automatic Detection of Microsleep Episodes with Deep Learning},
  author={Malafeev, Alexander and Hertig-Godeschalk, Annette and 
          Schreier, Dario R and Skorucak, Jan and Mathis, Johannes and 
          Achermann, Peter},
  journal={Frontiers in Neuroscience},
  volume={15},
  pages={564098},
  year={2021},
  publisher={Frontiers Media SA},
  doi={10.3389/fnins.2021.564098}
}
```

### Scope of Adaptation & Original Work

While this project uses the Malafeev et al. CNN architecture as a **methodological foundation**, it represents **substantial independent research and implementation**:

#### 1. Problem Reformulation

| Aspect | Malafeev et al. (Original) | This Project |
|--------|---------------------------|--------------|
| **Task** | Multi-class microsleep episode detection | Binary drowsiness classification |
| **Input** | 20-channel polysomnography (EEG/EOG/EMG) | 2-channel EOG only |
| **Output** | Episode classification (5 classes) | Binary (Awake vs. Drowsy) |
| **Context** | Clinical sleep laboratory | Real-world drowsiness monitoring |

#### 2. Complete Data Pipeline Redesign

- **Dataset**: Proprietary sleep lab data → Public Dryad drowsiness dataset
- **Preprocessing**: Entirely new pipeline for EOG-only extraction from multi-modal EDF files
- **Annotation parsing**: New labeling methodology with ±1s context windows
- **Data loading**: Custom `loadData.py` implementation for Dryad format
- **Channel mapping**: 20+ channels → 2 channels (architecture input adaptation)

#### 3. Technical Implementation from Scratch

- **Framework migration**: Original implementation → TensorFlow 2.x / Keras 3.x
- **Environment**: Local/cluster → Google Colab with Drive integration
- **Class imbalance handling**: Weighted loss functions, experimentation with SMOTE
- **Hyperparameter tuning**: Batch sizes, learning rates, epochs optimized for new dataset
- **Training infrastructure**: Complete training pipelines for 3 experiments

#### 4. Novel Experimental Contributions (Beyond Original Paper)

This project extends beyond the original work with **three comprehensive experiments**:

**Experiment 1 (Phase1_CNN)**:
- Reimplemented CNN architecture for drowsiness (vs. microsleep)
- 6 window size variants (2s, 4s, 8s, 16s, 32s + CNN-LSTM)
- Cross-subject validation with Dryad dataset

**Experiment 2 (Phase2_ML)** — **Entirely Novel**:
- 63 hand-crafted features from time/frequency/nonlinear domains
- Comparison of 6 ML algorithms (LR, RF, SVM, XGBoost, LightGBM, Ensemble)
- Interpretability analysis (feature importance)

**Experiment 3 (Alternative_Splits)** — **Entirely Novel**:
- Three data splitting strategies (Within-Subject, 10-fold CV, Split 1.5)
- Optimization of training data utilization
- Best results achieved (Kappa: 0.433)

#### 5. Extensive Evaluation Framework

- Cohen's Kappa as primary metric (appropriate for imbalanced data)
- Per-class metrics (precision, recall, F1 for both classes)
- Sensitivity, specificity, false positive/negative rates
- Confusion matrices with normalization
- Cross-experiment comprehensive comparison

### Dataset Citation

**Dryad Drowsiness Dataset**:

Chuang, C.-H., Huang, C.-S., Ko, L.-W., & Lin, C.-T. (2018). *An EEG-based perceptual function integration network for application to drowsy driving*. Dryad Digital Repository. https://doi.org/10.5061/dryad.5tb2rbp9c

```bibtex
@misc{chuang2018eeg,
  title={An EEG-based perceptual function integration network for 
         application to drowsy driving},
  author={Chuang, Chun-Hsiang and Huang, Chung-Song and 
          Ko, Li-Wei and Lin, Chin-Teng},
  year={2018},
  publisher={Dryad Digital Repository},
  doi={10.5061/dryad.5tb2rbp9c}
}
```

### Summary

**Architectural Inspiration**: Malafeev et al. (2021)  
**Implementation**: Entirely original  
**Dataset**: Different (Dryad vs. proprietary)  
**Task**: Different (drowsiness vs. microsleep)  
**Experiments 2 & 3**: Novel contributions  
**Preprocessing, training, evaluation**: Original work

This project demonstrates how a published CNN architecture can be successfully **adapted and extended** for a related but distinct problem (drowsiness detection from EOG-only signals), while contributing novel experimental approaches and comprehensive comparative analysis.


## 🤝 Contributing

Contributions are welcome! Potential improvements:

**Phase 1 Enhancements**:
- Additional CNN architectures (ResNet, Transformer)
- Cross-subject validation strategies
- Real-time detection pipeline
- Mobile deployment

**Phase 2 Enhancements**:
- Feature selection optimization
- Hyperparameter tuning (grid search)
- Additional ML models
- SHAP values for interpretability
- SMOTE for minority class oversampling

## 👥 Authors

**Original Microsleep Detection Method**: Malafeev et al. (2021)  
**Drowsiness Detection Adaptation**: Extensively modified and adapted for Dryad dataset with EOG-only binary classification

This implementation represents independent work adapting the microsleep detection architecture to drowsiness detection, requiring complete redesign of data preprocessing, model architecture, training procedures, and evaluation framework.

## 🙏 Acknowledgments

- **Malafeev et al.** for the original microsleep detection CNN architecture concept
- **Dryad dataset providers** for the drowsiness detection dataset
- **Open-source community** for TensorFlow, Keras, and scientific Python tools


**Note**: This project is for research and educational purposes. The system is not intended for clinical, safety-critical, or real-world drowsiness monitoring applications without extensive additional validation, testing, and regulatory approval.

**Research Focus**: This work demonstrates two complementary approaches to drowsiness detection from EOG signals - end-to-end deep learning vs. feature engineering - highlighting the trade-offs between performance, interpretability, and computational requirements.
