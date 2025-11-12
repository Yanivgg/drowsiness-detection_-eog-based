# Model Training Guide

Complete guide for training microsleep detection models locally or on Google Colab.

## Overview

This guide covers training 6 model variants:
- **CNN_2s, CNN_4s, CNN_8s, CNN_16s, CNN_32s**: CNN with different window sizes
- **CNN_LSTM**: Hybrid CNN-LSTM architecture

**Recommended**: CNN_16s (best performance)

## Prerequisites

- Preprocessed `.mat` files in `data/processed/files/`
- `data/file_sets.mat` (train/val/test split)
- GPU (highly recommended)
- 8+ GB RAM

## Training Options

### Option 1: Google Colab (Recommended)

**Pros**:
- Free GPU access (T4, ~16 GB)
- No local setup required
- ~10-20x faster than CPU

**Cons**:
- Session timeouts (need keep-alive)
- Google Drive dependency

### Option 2: Local Training

**Pros**:
- Full control
- No session limits

**Cons**:
- Requires GPU (very slow on CPU)
- Local environment setup

---

## Google Colab Training

### Step 1: Prepare Package

Package all necessary files:

```bash
python scripts/prepare_for_colab.py --auto
```

**Output**: `microsleep_colab_package.zip` (~200 MB)

**Package contents**:
- All `.mat` data files
- Model code (all 6 variants)
- Utility scripts
- Split configuration

### Step 2: Upload to Colab

1. Open Google Colab: https://colab.research.google.com
2. Upload `microsleep_colab_package.zip`
3. Upload `notebooks/microsleep_colab_complete.ipynb`

**Or**:
- Copy cells from `scripts/microsleep_colab_complete.py` manually

### Step 3: Setup Colab Environment

**Cell 1: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2: Extract Package**
```python
!unzip -q microsleep_colab_package.zip
!ls data/processed/files/  # Verify 20 files
```

**Cell 3: Install Dependencies**
```python
!pip install -q keras tensorflow scikit-learn scipy
```

**Cell 4: Verify GPU**
```python
import tensorflow as tf
print("GPU available:", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
```

**Expected**: `GPU available: 1` with Tesla T4

**If no GPU**: Runtime → Change runtime type → Hardware accelerator: GPU

### Step 4: Configure Training

```python
# ========================================
# CONFIGURATION
# ========================================

# Model selection
MODEL_TO_TRAIN = 'CNN_16s'  # Options: CNN_2s, CNN_4s, CNN_8s, CNN_16s, CNN_32s, CNN_LSTM

# Training parameters
EPOCHS = 3                   # Number of epochs (3-6 recommended)
BATCH_SIZE = 800            # Batch size (adjust for GPU memory)
STRIDE = 1                  # Window stride (1=max data augmentation)

# Paths
BASE_PATH = '/content'                                    # Colab working directory
DRIVE_PATH = '/content/drive/MyDrive/microsleep_results'  # Save location
```

**Parameter guidelines**:

| Parameter | Recommended | Alternative | Notes |
|-----------|-------------|-------------|-------|
| MODEL_TO_TRAIN | CNN_16s | CNN_8s | Best performance |
| EPOCHS | 3-6 | 10+ | More epochs may overfit |
| BATCH_SIZE | 800 | 400, 1600 | Depends on GPU memory |
| STRIDE | 1 | 200 | 1=max augmentation, slower |

### Step 5: Run Training

Execute the training cell:

```python
%cd /content/src/models/cnn/{MODEL_TO_TRAIN}/
!python train.py
```

**Training process**:
1. Loads training data
2. Calculates class weights
3. Creates data generator
4. Builds model
5. Trains for specified epochs
6. Saves model checkpoints
7. Evaluates on validation set

**Output per epoch**:
```
Epoch = 0
43530/43530 ━━━━━━━━━━━━━━━━━━━━ 3700s 84ms/step - accuracy: 0.9558 - loss: 0.0984
K val per class =  [0.06148454 0.06148454]
K val =  0.06148454129438907
loss val =  0.02152515295892954
```

**Training time** (CNN_16s on T4):
- Per epoch: ~1 hour
- 3 epochs: ~3 hours
- Full training: ~3-6 hours

### Step 6: Save Results

Models are automatically saved to:
```
{MODEL_DIR}/models/model_ep{N}.h5     # Model checkpoint per epoch
{MODEL_DIR}/model.h5                  # Final model
{MODEL_DIR}/predictions.mat           # Training history & predictions
```

**Copy to Google Drive**:
```python
import shutil

# Copy trained model
shutil.copy(f'./model.h5', f'{DRIVE_PATH}/{MODEL_TO_TRAIN}_model.h5')

# Copy predictions
shutil.copy(f'./predictions.mat', f'{DRIVE_PATH}/{MODEL_TO_TRAIN}_predictions.mat')

print(f"✓ Saved to {DRIVE_PATH}")
```

### Step 7: Run Predictions

**Validation set**:
```python
!python predict_val.py
```

**Test set**:
```python
!python predict.py
```

**Save test results**:
```python
# Results saved automatically to:
# {DRIVE_PATH}/{MODEL_TO_TRAIN}_test_results.mat
```

### Step 8: Keep Session Alive

Prevent Colab from disconnecting:

```javascript
// Run in browser console (F12)
function ClickConnect(){
    console.log("Clicking connect button");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)  // Click every minute
```

---

## Local Training

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Data

```bash
# Check data files
ls data/processed/files/  # Should see 20 .mat files

# Check split configuration
python -c "import scipy.io; print(scipy.io.loadmat('data/file_sets.mat').keys())"
```

### Step 3: Choose Model

```bash
cd src/models/cnn/CNN_16s  # Or CNN_2s, CNN_4s, etc.
```

### Step 4: Configure Training

Edit `train.py` parameters:

```python
# Training configuration
n_epochs = 3
batch_size = 200  # Smaller for CPU/low-memory GPU
prec = 1          # Stride
```

### Step 5: Run Training

```bash
python train.py
```

**Note**: CPU training is VERY slow (~hours per epoch). GPU highly recommended.

### Step 6: Monitor Progress

```bash
# Training outputs:
./models/model_ep0.h5, model_ep1.h5, ...  # Checkpoints
./model.h5                                 # Final model
./predictions.mat                          # Training history
./predictions/predictions_ep0.mat, ...     # Per-epoch predictions
```

### Step 7: Evaluate

```bash
# Validation set
python predict_val.py

# Test set  
python predict.py
```

---

## Training Parameters Explained

### Window Size (Model Selection)

Window size determines temporal context:

| Window | Samples | Context | Use Case |
|--------|---------|---------|----------|
| 2s | 400 | Minimal | Fast detection |
| 4s | 800 | Short | Quick events |
| 8s | 1,600 | Medium | Balanced |
| **16s** | **3,200** | **Long** | **Best overall** ⭐ |
| 32s | 6,400 | Very long | Maximum context |

**Recommendation**: Start with CNN_16s (best performance)

### Epochs

Number of complete passes through training data:

| Epochs | Training Time | Performance | Notes |
|--------|---------------|-------------|-------|
| 1-2 | ~1-2 hours | Poor | Undertrained |
| **3-6** | **~3-6 hours** | **Good** | **Recommended** ⭐ |
| 10+ | ~10+ hours | Marginal | May overfit |

**Signs of convergence**:
- Training loss decreases
- Validation Kappa stabilizes or decreases
- Validation loss stops improving

**Stop if**:
- Validation Kappa drops (overfitting)
- Validation loss increases
- No improvement for 2+ epochs

### Batch Size

Number of samples per gradient update:

| Batch Size | GPU Memory | Speed | Gradient Quality |
|------------|------------|-------|------------------|
| 200 | 4 GB | Slow | Noisy |
| 400 | 8 GB | Medium | Moderate |
| **800** | **16 GB** | **Fast** | **Good** ⭐ |
| 1600 | 32 GB | Very fast | Smooth |

**Colab T4**: Use 800  
**Local GPU (<8GB)**: Use 200-400  
**CPU**: Use 50-100

### Stride (Data Augmentation)

Step size for sliding window:

| Stride | Windows/File | Training Speed | Data Augmentation |
|--------|--------------|----------------|-------------------|
| **1** | **~2.9M** | **Slow** | **Maximum** ⭐ |
| 40 | ~73K | Medium | Moderate |
| 200 | ~15K | Fast | Minimal |
| 3200 | ~900 | Very fast | None |

**Recommendation**: 
- Training: stride=1 (maximum data)
- Quick experiments: stride=200

---

## Class Imbalance Handling

### Problem

Dataset is highly imbalanced:
- Awake: ~96.5%
- Drowsy: ~3.5%

**Without handling**: Model predicts all "Awake" (96.5% accuracy, useless)

### Solution: Class Weights

Automatically calculated and applied:

```python
# In train.py
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

print(f"Class weights: {class_weights}")
# Output: [0.507, 32.468]  # Drowsy samples weighted 64x more
```

**Applied during training**:
```python
# Generator yields sample weights
sample_weights = class_weights[labels]

# Used in model.fit()
model.fit(generator, ..., sample_weight=sample_weights)
```

### Verification

Check class weights in training output:

```
=====================
class weights 
[ 0.50617456 40.98870056]
=====================
```

**Good weights**: Drowsy class 20-50x higher than Awake

---

## Model Architecture Details

### CNN Models

```
Input: (batch, window_samples, 1, 2)  # 2 EOG channels
  ↓
Gaussian Noise (σ=0.1)  # Regularization
  ↓
12x [ Conv2D → BatchNorm → ReLU → MaxPool ]
  Filters: 32 → 64 → 128 → 128 → 128 → 256 → 256 → 256 → 256
  ↓
Flatten
  ↓
Dense(256) → BatchNorm → Dropout(0.25)
  ↓
Dense(256) → BatchNorm → Dropout(0.25)
  ↓
Dense(2, softmax)  # Binary output
  ↓
Output: (batch, 2)  # [P(Awake), P(Drowsy)]
```

**Total parameters**: ~1.34M (CNN_16s)

### CNN-LSTM

```
Input: (batch, seq_len, window_samples, 1, 2)
  ↓
TimeDistributed(CNN block)
  ↓
Bidirectional LSTM(128)
  ↓
TimeDistributed Dense(2, softmax)
  ↓
Output: (batch, seq_len, 2)
```

**Note**: CNN-LSTM requires more tuning and epochs

---

## Monitoring Training

### Training Metrics

Monitor in real-time:

```python
Epoch = 0
50813/50813 ━━━━━━━━━━━━━━━━━━━━ 12240s 240ms/step
- accuracy: 0.9605    # Training accuracy
- loss: 0.0910        # Training loss
```

**What to watch**:
- ✓ Loss decreasing (learning)
- ✓ Accuracy increasing
- ✗ Loss stuck (not learning)
- ✗ Accuracy = class ratio (predicting all one class)

### Validation Metrics

Per epoch validation:

```
K val per class = [0.08485091 0.08485091]
K val = 0.08485091242316534
loss val = 0.021378048152352374
```

**What to watch**:
- ✓ Kappa increasing (performance improving)
- ✓ Validation loss decreasing
- ✗ Kappa decreasing (overfitting)
- ✗ Validation loss increasing (overfitting)

### Training Curves

Plot after training:

```python
import scipy.io as spio
import matplotlib.pyplot as plt

# Load training history
history = spio.loadmat('predictions.mat')

# Plot loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss_tr'].flatten(), label='Training')
plt.plot(history['loss_val'].flatten(), label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training/Validation Loss')

# Plot kappa
plt.subplot(1, 2, 2)
plt.plot(history['acc_val'].flatten())
plt.xlabel('Epoch')
plt.ylabel('Validation Kappa')
plt.title('Validation Performance')

plt.tight_layout()
plt.show()
```

---

## Troubleshooting

### GPU Out of Memory

**Error**: `ResourceExhaustedError: OOM when allocating tensor`

**Solutions**:
1. Reduce batch size: `BATCH_SIZE = 400`
2. Clear GPU cache: `tf.keras.backend.clear_session()`
3. Restart Colab runtime
4. Use smaller model (CNN_8s instead of CNN_16s)

### Training Too Slow

**Issue**: Hours per epoch on CPU

**Solutions**:
1. Use Google Colab GPU (free)
2. Increase stride: `prec = 200`
3. Reduce batch size: `batch_size = 100`

### Model Not Learning

**Symptoms**: Loss not decreasing, accuracy stuck at ~96%

**Causes & Solutions**:
1. **Class weights not applied**:
   - Check generator yields sample_weights
   - Verify class_weights printed during training

2. **Learning rate too high/low**:
   ```python
   from keras.optimizers import Nadam
   optimizer = Nadam(learning_rate=0.0005)  # Adjust
   ```

3. **Data loading issue**:
   - Verify file_sets.mat exists
   - Check data shapes match model input

### Validation Kappa Very Low

**Issue**: Validation Kappa < 0.1 while training looks good

**Explanation**: Normal! Validation subjects may differ from training distribution

**Solutions**:
- Focus on test set Kappa (more representative)
- Train for more epochs
- Check validation subject characteristics

### Colab Session Disconnects

**Issue**: Training interrupted after 1-2 hours

**Solutions**:
1. Use keep-alive script (see Step 8 above)
2. Save checkpoints frequently (automatic)
3. Enable notifications when training completes
4. Use Colab Pro for longer sessions

---

## After Training

### 1. Save Everything

```python
# Models
shutil.copy('model.h5', f'{DRIVE_PATH}/{MODEL}_model.h5')

# Training history
shutil.copy('predictions.mat', f'{DRIVE_PATH}/{MODEL}_history.mat')

# Test results
shutil.copy('test_results.mat', f'{DRIVE_PATH}/{MODEL}_test_results.mat')
```

### 2. Evaluate Performance

Run comprehensive analysis:

```python
# In notebooks/results_analysis.ipynb
# Loads all models and generates comparison
```

### 3. Compare Models

See which performed best:

```bash
# In analysis notebook
# Compares Kappa, Precision, Recall across all models
```

### 4. Next Steps

- **Best model**: Use for further experiments
- **Ensemble**: Combine multiple models
- **Fine-tune**: Train for more epochs
- **Feature engineering**: Extract features for traditional ML

---

## Tips for Best Results

1. **Start with CNN_16s**: Best documented performance
2. **Use GPU**: 10-20x faster than CPU
3. **Monitor training**: Watch for overfitting
4. **Save checkpoints**: Don't lose progress
5. **Test multiple models**: Compare different window sizes
6. **Be patient**: Quality training takes hours
7. **Document settings**: Record all hyperparameters
8. **Verify data**: Ensure preprocessing correct

---

## Next Steps

After training:

1. **Evaluate models**: See [RESULTS_ANALYSIS_GUIDE.md](RESULTS_ANALYSIS_GUIDE.md)
2. **Compare performance**: Run analysis notebook
3. **Select best model**: Based on test Kappa
4. **Deploy/experiment**: Use for applications

---

## References

- Original paper: Malafeev et al. (2021) [10.3389/fnins.2021.564098](https://doi.org/10.3389/fnins.2021.564098)
- TensorFlow documentation: https://www.tensorflow.org/
- Keras documentation: https://keras.io/

