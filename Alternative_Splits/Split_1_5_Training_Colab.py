# %% [markdown]
# # 1.5 Recordings Split - Training Script
# 
# This script trains CNN_16s model with 1.5 recordings per subject:
# - **Train**: All _1 recordings (full) + First 50% of _2 recordings
# - **Test**: Last 50% of _2 recordings
# 
# **Advantages**:
# - 50% more training data than within-subject split
# - Tests temporal generalization (unseen second half)
# - Single training run (~2 hours)
# - No modifications to existing Phase1_CNN code

# %% [markdown]
# ## Setup and Imports

# %%
import os
import sys
import math
import random
import numpy as np
import scipy.io as spio
import keras
from keras.layers import concatenate
from sklearn.metrics import cohen_kappa_score
from keras import optimizers
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Layer, Dense, Dropout, Input, Activation, TimeDistributed, Reshape
from keras.layers import GRU, Bidirectional
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LSTM, ZeroPadding2D, GlobalAveragePooling2D
from keras.callbacks import History
from keras.models import Model
from collections import Counter
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

np.random.seed(0)

print("="*80)
print("1.5 Recordings Split Training Pipeline")
print("="*80)

# %% [markdown]
# ## Configuration

# %%
# ========================================
# CONFIGURATION - MODIFY AS NEEDED
# ========================================

# Base paths
DRIVE_BASE = '/content/drive/MyDrive/drowsiness-detection_-eog-based'
DATA_DIR = f'{DRIVE_BASE}/Phase1_CNN/data/processed/files/'
PHASE1_SRC = f'{DRIVE_BASE}/Phase1_CNN/src'
RESULTS_DIR = f'{DRIVE_BASE}/alternative_splits_results/split_1_5'

# Training parameters
BATCH_SIZE = 800
EPOCHS = 5
STRIDE = 1

# Model parameters (CNN_16s)
FS = 200  # sampling rate
W_LEN = 8 * FS  # half window: 8 seconds * 200 Hz = 1600 samples
DATA_DIM = W_LEN * 2  # full window: 3200 samples (16 seconds)
N_CL = 2  # binary classification
N_CHANNELS = 2  # 2 EOG channels

print(f"Configuration loaded")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Stride: {STRIDE}")
print(f"  Window: {DATA_DIM} samples (16 seconds)")

# %% [markdown]
# ## Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print("Google Drive mounted")

# %% [markdown]
# ## Verify Files

# %%
print("="*80)
print("Verifying files...")
print("="*80)

# Check data directory
if os.path.exists(DATA_DIR):
    mat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.mat')]
    print(f"\n✓ Data directory found with {len(mat_files)} .mat files")
    if len(mat_files) == 20:
        print("  ✓ All 20 files present!")
    else:
        print(f"  ⚠️ Expected 20 files, found {len(mat_files)}")
else:
    print(f"\n❌ Data directory not found: {DATA_DIR}")
    raise FileNotFoundError(f"Data directory not found")

# Check Phase1_CNN source
if os.path.exists(PHASE1_SRC):
    print(f"✓ Phase1_CNN source found: {PHASE1_SRC}")
else:
    print(f"❌ Phase1_CNN source not found: {PHASE1_SRC}")
    raise FileNotFoundError(f"Phase1_CNN source not found")

print("\n" + "="*80)
print("✓ All files verified")
print("="*80)

# %% [markdown]
# ## GPU Check

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
# ## Import Phase1_CNN Model

# %%
# Add Phase1_CNN to path
sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, f'{PHASE1_SRC}/models/cnn/CNN_16s')

from myModel import build_model

print("✓ Successfully imported Phase1_CNN model")

# %% [markdown]
# ## Custom Data Loading Functions
# 
# These functions are copied and modified from Phase1_CNN to support 50% splits

# %%
def load_recording_split(dir_name, f_name, split_type='full', n_cl=2):
    """
    Load a single .mat recording file with optional 50% split
    
    Args:
        dir_name: directory containing the .mat file
        f_name: filename of the .mat file
        split_type: 'full', 'first_half', 'second_half'
        n_cl: number of classes (default 2 for binary)
        
    Returns:
        tuple: (E1, E2, EOG, targets_O1, targets_O2)
    """
    mat = spio.loadmat(dir_name + f_name, struct_as_record=False, squeeze_me=True)
    
    # Load labels
    labels_O1 = mat['Data'].labels_O1
    labels_O2 = mat['Data'].labels_O2
    
    # Load EOG channels
    LOC = mat['Data'].E2  # Left EOG
    ROC = mat['Data'].E1  # Right EOG
    
    # Apply split if needed (only for _2 files)
    if split_type != 'full' and f_name.endswith('_2.mat'):
        split_point = len(labels_O1) // 2
        
        if split_type == 'first_half':
            labels_O1 = labels_O1[:split_point]
            labels_O2 = labels_O2[:split_point]
            LOC = LOC[:split_point]
            ROC = ROC[:split_point]
        elif split_type == 'second_half':
            labels_O1 = labels_O1[split_point:]
            labels_O2 = labels_O2[split_point:]
            LOC = LOC[split_point:]
            ROC = ROC[split_point:]
    
    # Expand dimensions to match expected shape
    LOC = np.expand_dims(LOC, axis=-1)
    LOC = np.expand_dims(LOC, axis=-1)
    ROC = np.expand_dims(ROC, axis=-1)
    ROC = np.expand_dims(ROC, axis=-1)
    
    # Concatenate to create EOG with 2 channels
    EOG = np.concatenate((LOC, ROC), axis=2)
    
    targets_O1 = labels_O1
    targets_O2 = labels_O2
    
    return (ROC, LOC, EOG, targets_O1, targets_O2)


def classes_split(dir_name, f_name, split_type='full'):
    """
    Get class labels from a recording with optional split
    
    Args:
        dir_name: directory containing the .mat file
        f_name: filename of the .mat file
        split_type: 'full', 'first_half', 'second_half'
        
    Returns:
        array: labels from the recording
    """
    mat = spio.loadmat(dir_name + f_name, struct_as_record=False, squeeze_me=True)
    labels_O1 = mat['Data'].labels_O1
    
    # Apply split if needed (only for _2 files)
    if split_type != 'full' and f_name.endswith('_2.mat'):
        split_point = len(labels_O1) // 2
        
        if split_type == 'first_half':
            labels_O1 = labels_O1[:split_point]
        elif split_type == 'second_half':
            labels_O1 = labels_O1[split_point:]
    
    return labels_O1


def classes_global_split(data_dir, files_train, split_type='full'):
    """
    Get all labels from training files for class weight computation
    
    Args:
        data_dir: directory containing .mat files
        files_train: list of training filenames
        split_type: 'full', 'first_half', 'second_half'
        
    Returns:
        list: all labels concatenated from all training files
    """
    print("=====================")
    print(f"Reading train set for class distribution (split_type={split_type}):")
    st0 = []
    
    for f in files_train:
        st = classes_split(data_dir, f, split_type)
        st0.extend(st)
    
    return st0


def load_data_split(data_dir, files_train, w_len, split_type='full'):
    """
    Load multiple recordings with padding for sliding window and optional split
    
    Args:
        data_dir: directory containing .mat files
        files_train: list of filenames to load
        w_len: half window length in samples
        split_type: 'full', 'first_half', 'second_half'
        
    Returns:
        tuple: (data_train, targets_train, N_samples)
    """
    E1, E2, X2, targets1, targets2 = load_recording_split(data_dir, files_train[0], split_type)
    print(E1.shape)
    
    data_train = []
    targets_train = []
    
    # Padding for sliding windows
    padding = ((w_len, w_len), (0, 0), (0, 0))
    
    N_samples = 0
    for f in files_train:
        print(f)
        E1, E2, X2, targets1, targets2 = load_recording_split(data_dir, f, split_type)
        
        # Apply padding
        E1 = np.pad(E1, pad_width=padding, mode='constant', constant_values=0)
        E2 = np.pad(E2, pad_width=padding, mode='constant', constant_values=0)
        X2 = np.pad(X2, pad_width=padding, mode='constant', constant_values=0)
        
        l = targets1.shape[0]
        N_samples += 2 * len(targets1)
        
        data_train.append([E1, E2, X2])
        targets_train.append([targets1, targets2])
    
    return (data_train, targets_train, N_samples)


def batch_generator(sample_list, batch_size):
    """
    Generate batches from a list of samples
    
    Args:
        sample_list: list of samples
        batch_size: size of each batch
        
    Yields:
        batch: list of samples in current batch
    """
    for i in range(0, len(sample_list), batch_size):
        yield sample_list[i:i + batch_size]

print("✓ Custom data loading functions defined")

# %% [markdown]
# ## Define File Lists

# %%
print("="*80)
print("1.5 Recordings Split")
print("="*80)
print("\nFor each subject:")
print("  - Recording _1: Full → TRAIN")
print("  - Recording _2: First 50% → TRAIN, Last 50% → TEST")
print("="*80)

all_subjects = ['01M', '02F', '03F', '04M', '05M', '06M', '07F', '08M', '09M', '10M']

# Files for training
files_1_full = [f"{s}_1.mat" for s in all_subjects]  # All _1 recordings (full)
files_2_train = [f"{s}_2.mat" for s in all_subjects]  # _2 recordings (first half only)

# Files for testing
files_2_test = [f"{s}_2.mat" for s in all_subjects]  # _2 recordings (second half only)

print(f"\nTrain files (_1 full): {len(files_1_full)}")
for f in files_1_full[:3]:
    print(f"  - {f}")
print("  ...")

print(f"\nTrain files (_2 first 50%): {len(files_2_train)}")
for f in files_2_train[:3]:
    print(f"  - {f} (first half)")
print("  ...")

print(f"\nTest files (_2 last 50%): {len(files_2_test)}")
for f in files_2_test[:3]:
    print(f"  - {f} (second half)")
print("  ...")

# %% [markdown]
# ## Calculate Class Weights

# %%
print("\n" + "="*80)
print("Calculating class weights...")
print("="*80)

# Get labels from full _1 recordings
st0 = classes_global_split(DATA_DIR, files_1_full, split_type='full')

# Get labels from first half of _2 recordings
st0_half = classes_global_split(DATA_DIR, files_2_train, split_type='first_half')
st0.extend(st0_half)

cls = np.arange(N_CL)
cl_w = class_weight.compute_class_weight('balanced', classes=cls, y=st0)
print(f"Class weights: {cl_w}")

# %% [markdown]
# ## Load Training Data

# %%
print("\n" + "="*80)
print("Loading training data...")
print("="*80)

# Load full _1 recordings
print("\nLoading _1 recordings (full):")
(data_1, targets_1, N_samples_1) = load_data_split(DATA_DIR, files_1_full, W_LEN, split_type='full')
print(f'N_samples from _1 files: {N_samples_1}')

# Load first half of _2 recordings
print("\nLoading _2 recordings (first 50%):")
(data_2, targets_2, N_samples_2) = load_data_split(DATA_DIR, files_2_train, W_LEN, split_type='first_half')
print(f'N_samples from _2 first half: {N_samples_2}')

# Combine training data
data_train = data_1 + data_2
targets_train = targets_1 + targets_2
N_samples_train = N_samples_1 + N_samples_2

print(f'\nTotal training samples: {N_samples_train}')
N_batches = int(math.ceil((N_samples_train + 0.0) / BATCH_SIZE))
print(f'N_batches: {N_batches}')

# %% [markdown]
# ## Load Test Data

# %%
print("\n" + "="*80)
print("Loading test data...")
print("="*80)

print("\nLoading _2 recordings (last 50%):")
(data_test, targets_test, N_samples_test) = load_data_split(DATA_DIR, files_2_test, W_LEN, split_type='second_half')
print(f'N_samples_test: {N_samples_test}')

# %% [markdown]
# ## Create Sample Lists

# %%
# Create sample lists for training
sample_list_train = []
for i in range(len(targets_train)):
    for j in range(len(targets_train[i][0])):
        mid = j * STRIDE
        mid += W_LEN
        wnd_begin = mid - W_LEN
        wnd_end = mid + W_LEN - 1
        sample_list_train.append([i, j, wnd_begin, wnd_end])

# Create sample lists for testing
sample_list_test = []
for i in range(len(targets_test)):
    sample_list_test.append([])
    for j in range(len(targets_test[i][0])):
        mid = j * STRIDE
        mid += W_LEN
        wnd_begin = mid - W_LEN
        wnd_end = mid + W_LEN - 1
        sample_list_test[i].append([i, j, wnd_begin, wnd_end])

print(f"Training samples: {len(sample_list_train)}")
print(f"Test samples: {sum(len(s) for s in sample_list_test)}")

# %% [markdown]
# ## Define Data Generator

# %%
def my_generator(data, targets, sample_list, shuffle=True, class_weights=None):
    """Generator for training/validation data"""
    if shuffle:
        random.shuffle(sample_list)
    while True:
        for batch in batch_generator(sample_list, BATCH_SIZE):
            batch_data = []
            batch_targets = []
            batch_sample_weights = []
            for sample in batch:
                [f, s, b, e] = sample
                sample_label = targets[f][0][s]
                sample_x1 = data[f][0][b:e+1]  # E1
                sample_x2 = data[f][1][b:e+1]  # E2
                sample_x = np.concatenate((sample_x1, sample_x2), axis=2)
                batch_data.append(sample_x)
                batch_targets.append(sample_label)
                if class_weights is not None:
                    batch_sample_weights.append(class_weights[int(sample_label)])
            batch_data = np.stack(batch_data, axis=0)
            batch_targets = np.array(batch_targets)
            if class_weights is not None:
                batch_sample_weights = np.array(batch_sample_weights)
            batch_targets = to_categorical(batch_targets, N_CL)
            # Normalization
            batch_data = batch_data / 100
            batch_data = np.clip(batch_data, -1, 1)
            if class_weights is not None:
                yield batch_data, batch_targets, batch_sample_weights
            else:
                yield batch_data, batch_targets

print("✓ Generator defined")

# %% [markdown]
# ## Build Model

# %%
print("\n" + "="*80)
print("Building model...")
print("="*80)

ordering = 'channels_last'
keras.backend.set_image_data_format(ordering)

[cnn_eeg, model] = build_model(DATA_DIM, N_CHANNELS, N_CL)
Nadam = optimizers.Nadam()
model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# %% [markdown]
# ## Create Results Directory

# %%
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/models', exist_ok=True)
print(f"✓ Results directory: {RESULTS_DIR}")

# %% [markdown]
# ## Training Loop

# %%
print("\n" + "="*80)
print("Training model...")
print("="*80)

history = History()
acc_tr = []
loss_tr = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    generator_train = my_generator(data_train, targets_train, sample_list_train, class_weights=cl_w)
    model.fit(generator_train, steps_per_epoch=N_batches, epochs=1, verbose=1, callbacks=[history])
    
    acc_tr.append(history.history['accuracy'])
    loss_tr.append(history.history['loss'])
    
    # Save model
    model.save(f'{RESULTS_DIR}/models/model_ep{epoch}.h5')
    print(f"✓ Model saved: epoch {epoch}")

# Save final model
model.save(f'{RESULTS_DIR}/best_model.h5')
print("\n✓ Training complete!")

# %% [markdown]
# ## Evaluate on Test Set

# %%
print("\n" + "="*80)
print("Evaluating on test set...")
print("="*80)

test_y_pred = []
test_y_true = []

for j in range(len(data_test)):
    generator_test = my_generator(data_test, targets_test, sample_list_test[j], shuffle=False)
    y_pred = model.predict(generator_test, steps=int(math.ceil((len(sample_list_test[j]) + 0.0) / BATCH_SIZE)))
    test_y_pred.extend(np.argmax(y_pred, axis=1).flatten())
    test_y_true.extend(targets_test[j][0])

test_y_pred = np.array(test_y_pred)
test_y_true = np.array(test_y_true)

# %% [markdown]
# ## Calculate Metrics

# %%
accuracy = accuracy_score(test_y_true, test_y_pred)
precision = precision_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
recall = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
f1 = f1_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
kappa = cohen_kappa_score(test_y_true, test_y_pred)
cm = confusion_matrix(test_y_true, test_y_pred)

# Print results
print("\nTest Set Results:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  Cohen's Kappa: {kappa:.4f}")
print(f"\nConfusion Matrix:")
print(cm)

# %% [markdown]
# ## Save Results

# %%
results = {
    'y_true': test_y_true,
    'y_pred': test_y_pred,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'kappa': kappa,
    'confusion_matrix': cm,
    'train_files_1_full': files_1_full,
    'train_files_2_half': files_2_train,
    'test_files_2_half': files_2_test,
    'split_method': '1.5_recordings_per_subject'
}

spio.savemat(f'{RESULTS_DIR}/test_results.mat', results)
print(f"\n✓ Results saved to {RESULTS_DIR}")

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*80)
print("1.5 RECORDINGS SPLIT - SUMMARY")
print("="*80)
print(f"\nModel: CNN_16s")
print(f"Train: {len(files_1_full)} full _1 recordings + {len(files_2_train)} half _2 recordings")
print(f"       = 1.5 recordings per subject")
print(f"Test:  {len(files_2_test)} half _2 recordings (last 50%)")
print(f"       = 0.5 recordings per subject")
print(f"\nTest Set Performance:")
print(f"  Cohen's Kappa: {kappa:.3f}")
print(f"  Precision: {precision*100:.1f}%")
print(f"  Recall: {recall*100:.1f}%")
print(f"  F1-Score: {f1:.3f}")
print("\n" + "="*80)
print("Training Complete!")
print("="*80)

# %%
print("="*80)
print("Results saved to Google Drive")
print("="*80)
print(f"\nLocation: {RESULTS_DIR}")
print("\nYou can access them from your Google Drive!")

# %%

