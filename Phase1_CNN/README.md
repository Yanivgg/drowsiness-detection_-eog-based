# Phase 1: CNN-Based Drowsiness Detection

## Overview

Phase 1 implements an **end-to-end deep learning approach** for drowsiness detection using Convolutional Neural Networks (CNNs). This phase adapts the Malafeev et al. (2021) microsleep detection architecture for drowsiness detection on the Dryad dataset.

### Key Features

- **End-to-end learning**: No manual feature engineering required
- **6 model variants**: Different temporal window sizes (2s, 4s, 8s, 16s, 32s) + CNN-LSTM
- **Best performance**: CNN_16s achieves Cohen's Kappa = 0.394
- **Google Colab ready**: Complete GPU training pipeline
- **Class imbalance handling**: Custom weighting system for 96.5% vs 3.5% split

## 📁 Directory Structure

```
Phase1_CNN/
├── README.md                          # This file
├── data/                              # Processed data
│   ├── processed/files/               # 20 .mat files (240 MB)
│   ├── file_sets.mat                 # Train/val/test split
│   └── create_file_sets.py           # Split configuration
├── src/                               # Source code
│   ├── loadData.py                   # Data loading utilities
│   ├── utils.py                      # Helper functions
│   └── models/                       # Model architectures
│       ├── cnn/                      # CNN variants
│       │   ├── CNN_2s/               # 2-second window
│       │   ├── CNN_4s/               # 4-second window
│       │   ├── CNN_8s/               # 8-second window
│       │   ├── CNN_16s/              # 16-second window (best)
│       │   └── CNN_32s/              # 32-second window
│       └── cnn_lstm/                 # CNN-LSTM hybrid
│           └── CNN_LSTM/
├── preprocessing/                     # Data preprocessing
│   ├── edf_to_mat_eog_only.py       # EDF → MAT conversion
│   ├── eda.py                        # Exploratory data analysis
│   ├── annotations_summary.csv       # Event distribution
│   ├── dryad_eda_summary.csv        # Dataset statistics
│   └── README.md
├── notebooks/
│   ├── CNN_Training_Colab.ipynb     # Training notebook
│   └── CNN_Results_Analysis.ipynb    # Results analysis
└── scripts/
    ├── cnn_training_colab.py         # Training script (.py with cell markers)
    ├── cnn_results_analysis.py       # Analysis script (.py with cell markers)
    ├── data_exploration.py           # EDA script (.py with cell markers)
    └── prepare_for_colab.py          # Package for Colab upload
```

## 🚀 Quick Start

### Step 1: Data Preprocessing (If Needed)

If you have raw EDF files:

```bash
cd preprocessing
python edf_to_mat_eog_only.py
```

**Output**: `data/processed/files/*.mat` (20 files, ~240 MB)

### Step 2: Package for Colab

```bash
cd scripts
python prepare_for_colab.py --auto
```

**Output**: `microsleep_colab_package.zip` (~200 MB)

### Step 3: Train on Google Colab

1. Upload `microsleep_colab_package.zip` to Colab
2. Upload `notebooks/CNN_Training_Colab.ipynb`
3. Run training cells
4. Models auto-save to Google Drive

### Step 4: Analyze Results

```bash
# In Jupyter or Colab
jupyter notebook notebooks/CNN_Results_Analysis.ipynb

# Or as Python script
python scripts/cnn_results_analysis.py
```

## 🏗️ Model Architectures

### CNN Variants (2s, 4s, 8s, 16s, 32s)

Deep convolutional neural network with:

**Input**: EOG signals (2 channels × window_size samples)

**Architecture**:
- 12 Conv2D layers (32→256 filters)
- Batch Normalization + ReLU activation
- MaxPooling2D (stride 2)
- Gaussian Noise (σ=0.1) for regularization
- 2 Dense layers (256→2)
- Dropout (0.25)

**Output**: Binary classification (Awake/Drowsy)

**Window sizes**:
- CNN_2s: 400 samples (2 sec × 200 Hz)
- CNN_4s: 800 samples (4 sec)
- CNN_8s: 1,600 samples (8 sec)
- **CNN_16s**: 3,200 samples (16 sec) ⭐ **Best**
- CNN_32s: 6,400 samples (32 sec)

### CNN-LSTM

Hybrid architecture combining:
- **CNN block**: Feature extraction from EOG signals
- **Bidirectional LSTM**: Temporal sequence modeling (128 units)
- **TimeDistributed Dense**: Per-timestep classification

## 📈 Results Summary

### Model Comparison (Test Set: Subject 07)

| Model | Kappa ⭐ | Precision | Recall | Accuracy | TP Count |
|-------|---------|-----------|--------|----------|----------|
| **CNN_16s** 🏆 | **0.394** | **54.9%** | **32.7%** | 96.8% | **32,927** |
| CNN_8s | 0.289 | 58.9% | 20.2% | 96.8% | 20,362 |
| CNN_32s | 0.286 | 53.1% | 32.2% | 96.7% | 32,412 |
| CNN_4s | 0.046 | 19.7% | 3.2% | 96.2% | 3,222 |
| CNN_2s | 0.056 | 9.5% | 7.7% | 94.3% | 7,715 |
| CNN_LSTM | 0.008 | 4.2% | 4.6% | 93.1% | 4,582 |

### Key Insights

- **16-second window is optimal** for drowsiness detection
- Longer windows provide better temporal context
- CNN_LSTM underperforms (requires extensive hyperparameter tuning)
- **Precision-Recall trade-off**: CNN_8s has higher precision, CNN_16s has higher recall

### Performance Interpretation

**Cohen's Kappa**:
- 0.0-0.2: Poor agreement
- 0.2-0.4: Fair agreement ← CNN_16s/8s/32s
- 0.4-0.6: Moderate agreement
- 0.6+: Substantial agreement

**Best for different use cases**:
- **General use**: CNN_16s (best balance)
- **Minimize false alarms**: CNN_8s (higher precision)
- **Maximize detection**: CNN_16s (higher recall)

## 🔧 Training Configuration

### Recommended Settings (Colab)

```python
MODEL_TO_TRAIN = 'CNN_16s'  # Best model
EPOCHS = 3-6                # Number of epochs
BATCH_SIZE = 800           # For T4 GPU
STRIDE = 1                 # Maximum data augmentation
```

### Data Augmentation

- **Stride=1**: Maximum overlap, ~2.9M windows per subject
- **Stride=200**: 1-second steps, ~14K windows
- **Stride=3200**: Non-overlapping, ~900 windows

### Class Imbalance Handling

The dataset is highly imbalanced (~3.5% drowsy samples). Handled by:

- **Class weights**: Automatically calculated (drowsy weighted 30-40x)
- **Sample weights**: Per-sample weighting in generator
- **Cohen's Kappa**: Primary metric (handles imbalance better than accuracy)

## 📊 Training Process

### Google Colab Training (Recommended)

**Advantages**:
- Free GPU access (T4, ~16 GB)
- No local setup required
- ~10-20x faster than CPU
- Automatic model saving to Drive

**Steps**:

1. **Setup Environment**:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Extract Package**:
```python
!unzip -q microsleep_colab_package.zip
```

3. **Verify GPU**:
```python
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))
```

4. **Select Model & Train**:
```python
MODEL_TO_TRAIN = 'CNN_16s'
EPOCHS = 5
BATCH_SIZE = 800
# Run training cells
```

5. **Monitor Training**:
- Watch loss and accuracy curves
- Check class weights are applied
- Verify validation metrics

**Runtime**: ~30-90 minutes per model

### Local Training

```bash
cd src/models/cnn/CNN_16s
python train.py
```

**Requirements**:
- GPU (CUDA-compatible)
- 8+ GB RAM
- TensorFlow 2.x with GPU support

**Note**: CPU training is very slow (~hours per epoch).

## 📉 Evaluation Metrics

### Primary Metric: Cohen's Kappa

Cohen's Kappa is used as the primary metric because:
- Handles class imbalance better than accuracy
- Measures agreement beyond chance
- Ranges from -1 to 1 (0 = random, 1 = perfect)

### Additional Metrics

- **Precision**: Of predicted drowsy samples, how many are correct?
- **Recall**: Of actual drowsy samples, how many are detected?
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True positives, false positives, etc.
- **Accuracy**: Overall correctness (less meaningful for imbalanced data)

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
# Solution: Check paths are correct
# Ensure zip was extracted properly
# Verify file_sets.mat exists
```

**4. Class imbalance warnings**
```
# Normal behavior - class weights automatically handle this
# Check training output for class weight values
```

**5. Validation Kappa very low**
```
# Expected - validation subjects may differ from training distribution
# Focus on test set Kappa as primary metric
```

**6. Drive disconnects in Colab**
```python
# Solution: Remount drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Performance Tips

**Speed up training**:
- Increase batch size (if GPU allows)
- Use stride > 1 for faster data generation
- Train on Colab GPU (10-20x faster)

**Improve model**:
- Train for more epochs (5-10)
- Try data augmentation techniques
- Experiment with different window sizes
- Ensemble multiple models

## 📖 Detailed Guides

- **[preprocessing/README.md](preprocessing/README.md)**: Data preprocessing guide
- **[../README.md](../README.md)**: Main project documentation
- **[../Phase2_ML/README.md](../Phase2_ML/README.md)**: Phase 2 ML approach

## 🔬 Technical Details

### Data Loading

The `loadData.py` module handles:
- Loading `.mat` files
- Extracting EOG channels (E1, E2)
- Creating sliding windows
- Generating sample weights
- Batch generation for training

### Model Training

Each model's `train.py` script:
- Loads train/val data from `file_sets.mat`
- Calculates class weights
- Creates data generators with sample weights
- Compiles model with weighted loss
- Trains with validation monitoring
- Saves best model based on validation loss

### Model Evaluation

Each model's `predict.py` script:
- Loads test data (Subject 07)
- Generates predictions
- Calculates all metrics
- Saves results to `.mat` file
- Prints performance summary

## 📄 File Descriptions

### Source Files

- `src/loadData.py`: Data loading utilities, generator creation
- `src/utils.py`: Helper functions (metrics, plotting, etc.)
- `src/models/cnn/CNN_*/myModel.py`: Model architecture definition
- `src/models/cnn/CNN_*/train.py`: Training script
- `src/models/cnn/CNN_*/predict.py`: Test set evaluation
- `src/models/cnn/CNN_*/predict_val.py`: Validation set evaluation

### Preprocessing Files

- `preprocessing/edf_to_mat_eog_only.py`: EDF → MAT conversion
- `preprocessing/eda.py`: Exploratory data analysis
- `preprocessing/annotations_summary.csv`: Drowsiness event statistics
- `preprocessing/dryad_eda_summary.csv`: Dataset overview

### Scripts

- `scripts/prepare_for_colab.py`: Package files for Colab upload
- `scripts/cnn_training_colab.py`: Training script with cell markers
- `scripts/cnn_results_analysis.py`: Results analysis script
- `scripts/data_exploration.py`: EDA script with cell markers

## 🎯 Next Steps

After completing Phase 1:

1. **Analyze Results**: Run `CNN_Results_Analysis.ipynb`
2. **Compare Models**: Evaluate window size impact
3. **Move to Phase 2**: Try feature engineering approach
4. **Compare Phases**: CNN vs ML performance

## 📞 Support

For issues or questions:
- Check troubleshooting section above
- Review main project README
- Check preprocessing README
- Open an issue on GitHub

---

**Phase 1 Focus**: End-to-end deep learning for drowsiness detection, demonstrating the power of CNNs to learn complex patterns directly from raw EOG signals without manual feature engineering.

