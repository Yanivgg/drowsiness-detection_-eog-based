# %% [markdown]
# # 🧠 Microsleep Detection Pipeline - Complete Workflow
# 
# This notebook contains the complete pipeline for training and evaluating microsleep detection models.
# 
# **Dataset:** Dryad EOG signals (2 channels: LOC-Ref, ROC-Ref)  
# **Task:** Binary classification (Awake vs. Drowsy)  
# **Models:** CNN (2s/4s/8s/16s/32s) and CNN-LSTM  
# **Test Subject:** 07 (100,600 microsleep samples)

# %% [markdown]
# ## 📦 Setup and Installation

# %%
# Install required packages (if needed)
# !pip install tensorflow keras scipy scikit-learn

# Import libraries
import os
import sys
import shutil
from pathlib import Path

print("="*80)
print("🚀 Microsleep Detection Pipeline")
print("="*80)

# %% [markdown]
# ## 🔧 Configuration

# %%
# ========================================
# CONFIGURATION - MODIFY AS NEEDED
# ========================================

# Base path (choose one)
# Option 1: Files uploaded directly to Colab
BASE_PATH = '/content'

# Option 2: Files mounted from Google Drive (uncomment to use)
# BASE_PATH = '/content/drive/MyDrive/microsleep_cursor'

# Model to train (choose one)
# Options: 'CNN_2s', 'CNN_4s', 'CNN_8s', 'CNN_16s', 'CNN_32s', 'CNN_LSTM'
MODEL_TO_TRAIN = 'CNN_8s'

# Training parameters
EPOCHS = 3
BATCH_SIZE = 800  # Increase for faster training (requires more GPU memory)
STRIDE = 1  # prec parameter (1=max augmentation, higher=faster but less data)

# Results directory
RESULTS_PATH = '/content/drive/MyDrive/microsleep_results'

# Save models to Drive?
SAVE_TO_DRIVE = True

print(f"✓ Configuration loaded")
print(f"  Base path: {BASE_PATH}")
print(f"  Model: {MODEL_TO_TRAIN}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Stride: {STRIDE}")

# %% [markdown]
# ## 📁 Mount Google Drive (Optional)

# %%
# Uncomment if using Google Drive
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# print("✓ Google Drive mounted")

# %% [markdown]
# ## 📂 Verify Files

# %%
print("="*80)
print("Checking files...")
print("="*80)

# Check data files
data_files_path = f'{BASE_PATH}/data/files'
file_sets_path = f'{BASE_PATH}/data/file_sets.mat'

if os.path.exists(data_files_path):
    mat_files = [f for f in os.listdir(data_files_path) if f.endswith('.mat')]
    print(f"✓ Found {len(mat_files)} .mat files in data/files/")
else:
    print(f"❌ data/files/ not found at {data_files_path}")

if os.path.exists(file_sets_path):
    import scipy.io as spio
    fs = spio.loadmat(file_sets_path)
    train_files = [f[0] for f in fs['files_train'].flatten()]
    val_files = [f[0] for f in fs['files_val'].flatten()]
    test_files = [f[0] for f in fs['files_test'].flatten()]
    print(f"✓ Found file_sets.mat")
    print(f"  Train: {len(train_files)} files (subjects 01-06)")
    print(f"  Val:   {len(val_files)} files (subjects 08-10)")
    print(f"  Test:  {len(test_files)} files (subject 07)")
else:
    print(f"❌ file_sets.mat not found")

# Check model directory
if MODEL_TO_TRAIN == 'CNN_LSTM':
    model_dir = f'{BASE_PATH}/code/CNN_LSTM'
else:
    model_dir = f'{BASE_PATH}/code/CNN/{MODEL_TO_TRAIN}'

if os.path.exists(model_dir):
    print(f"✓ Model directory found: {model_dir}")
else:
    print(f"❌ Model directory not found: {model_dir}")

# %% [markdown]
# ## 🎮 Check GPU

# %%
import tensorflow as tf

print("="*80)
print("GPU Check")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU available: {len(gpus)} device(s)")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
else:
    print("⚠️ No GPU found - training will be slow!")

# %% [markdown]
# ## ⚙️ Update Training Parameters

# %%
# Update batch_size, epochs, and stride in train.py

os.chdir(model_dir)
print(f"Working directory: {os.getcwd()}")

# Read train.py
with open('train.py', 'r', encoding='utf-8') as f:
    train_content = f.read()

# Update batch_size
if 'batch_size = ' in train_content:
    import re
    train_content = re.sub(r'batch_size = \d+', f'batch_size = {BATCH_SIZE}', train_content)
    print(f"✓ Updated batch_size to {BATCH_SIZE}")

# Update prec (stride)
if 'prec = ' in train_content:
    train_content = re.sub(r'prec = \d+', f'prec = {STRIDE}', train_content)
    print(f"✓ Updated stride (prec) to {STRIDE}")

# Update n_epochs
if 'n_epochs = ' in train_content:
    train_content = re.sub(r'n_epochs = \d+', f'n_epochs = {EPOCHS}', train_content)
    print(f"✓ Updated epochs to {EPOCHS}")

# Write back
with open('train.py', 'w', encoding='utf-8') as f:
    f.write(train_content)

print("✓ Training parameters updated")

# %% [markdown]
# ## 🏋️ Train Model

# %%
print("="*80)
print(f"Starting training: {MODEL_TO_TRAIN}")
print("="*80)

# Run training
exec(open('train.py').read())

print("\n" + "="*80)
print("✓ Training completed!")
print("="*80)

# %% [markdown]
# ## 💾 Save Model to Drive

# %%
if SAVE_TO_DRIVE:
    print("="*80)
    print("Saving model to Google Drive...")
    print("="*80)
    
    # Create results directory if needed
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Find the latest model
    model_files = []
    if os.path.exists('./models'):
        model_files = [f for f in os.listdir('./models') if f.endswith('.h5')]
    if os.path.exists('./model.h5'):
        model_files.append('model.h5')
    
    if model_files:
        # Copy model to Drive
        if os.path.exists('./model.h5'):
            src = './model.h5'
        elif model_files:
            # Get latest epoch model
            model_files_full = [f'./models/{f}' for f in model_files if 'models/' not in f]
            src = max(model_files_full, key=os.path.getmtime) if model_files_full else f'./models/{model_files[0]}'
        
        dst = f'{RESULTS_PATH}/{MODEL_TO_TRAIN}_model.h5'
        shutil.copy(src, dst)
        print(f"✓ Model saved to: {dst}")
        
        # Also save to local model.h5 for predict scripts
        if not os.path.exists('./model.h5'):
            shutil.copy(src, './model.h5')
            print(f"✓ Model copied to ./model.h5")
    else:
        print("⚠️ No model file found to save")
else:
    print("Model saving to Drive skipped (SAVE_TO_DRIVE=False)")

# %% [markdown]
# ## 📊 Validate on Validation Set

# %%
print("="*80)
print("Running validation...")
print("="*80)

# Run validation
exec(open('predict_val.py').read())

print("\n✓ Validation completed!")

# %% [markdown]
# ## 🧪 Predict on Test Set

# %%
print("="*80)
print("Running predictions on test set...")
print("="*80)

# Run prediction
exec(open('predict.py').read())

print("\n✓ Test predictions completed!")

# %% [markdown]
# ## 📈 Analyze Results

# %%
import scipy.io as spio
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

print("="*80)
print(f"TEST SET RESULTS - {MODEL_TO_TRAIN}")
print("="*80)

# Load predictions
pred_file = './predictions_output.mat'
data = spio.loadmat(pred_file)

y_true = data['y_true'].flatten()
y_pred = data['y_pred'].flatten()

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
kappa = cohen_kappa_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# Per-class metrics
if cm[1,0] + cm[1,1] > 0:
    recall_drowsy = cm[1,1] / (cm[1,0] + cm[1,1])
    precision_drowsy = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
else:
    recall_drowsy = 0
    precision_drowsy = 0

print(f"\n📊 Performance Metrics:")
print(f"  Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  F1-score:       {f1:.4f}")
print(f"  Cohen's Kappa:  {kappa:.4f}")

print(f"\n🎯 Microsleep Detection (Class 1):")
print(f"  Precision: {precision_drowsy:.4f} ({precision_drowsy*100:.2f}%)")
print(f"  Recall:    {recall_drowsy:.4f} ({recall_drowsy*100:.2f}%)")

print(f"\n📉 Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Awake    Drowsy")
print(f"Actual  Awake   {cm[0,0]:8,}  {cm[0,1]:6,}")
print(f"        Drowsy  {cm[1,0]:8,}  {cm[1,1]:6,}")

# Class distribution
n_awake = np.sum(y_true == 0)
n_drowsy = np.sum(y_true == 1)
print(f"\n📊 Class Distribution:")
print(f"  Awake:  {n_awake:,} ({n_awake/len(y_true)*100:.2f}%)")
print(f"  Drowsy: {n_drowsy:,} ({n_drowsy/len(y_true)*100:.2f}%)")

# Save summary
if SAVE_TO_DRIVE:
    summary = {
        'model': MODEL_TO_TRAIN,
        'accuracy': accuracy,
        'f1_score': f1,
        'kappa': kappa,
        'recall_drowsy': recall_drowsy,
        'precision_drowsy': precision_drowsy,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }
    summary_path = f'{RESULTS_PATH}/{MODEL_TO_TRAIN}_test_results.mat'
    spio.savemat(summary_path, summary)
    print(f"\n✓ Results saved to: {summary_path}")

print("="*80)

# %% [markdown]
# ## 🔄 Keep Colab Alive (Optional)

# %%
# Uncomment to prevent Colab from disconnecting due to inactivity

# from IPython.display import display, HTML
# 
# js_code = """
# <script>
# function KeepClicking(){
#     console.log("Keeping Colab alive...");
#     document.querySelector("colab-connect-button").click();
# }
# setInterval(KeepClicking, 60000);  // Click every 60 seconds
# </script>
# """
# 
# display(HTML(js_code))
# print("✓ Keep-alive script activated")

# %% [markdown]
# ## 🎉 Pipeline Complete!
# 
# ### Next Steps:
# 
# 1. **Train other models:** Change `MODEL_TO_TRAIN` and rerun training cells
# 2. **Compare models:** Use the saved results in `microsleep_results/`
# 3. **Tune hyperparameters:** Adjust `BATCH_SIZE`, `STRIDE`, or model architecture
# 4. **Analyze failures:** Look at cases where the model incorrectly predicts microsleep
# 
# ### Saved Files:
# - Model: `{RESULTS_PATH}/{MODEL_TO_TRAIN}_model.h5`
# - Results: `{RESULTS_PATH}/{MODEL_TO_TRAIN}_test_results.mat`
# - Predictions: `{model_dir}/predictions_output.mat`

