# %% [markdown]
# # 🔬 Alternative Data Splits - Training Script (v2)
# 
# Based on the original Phase1_CNN training code
# 
# This script trains CNN_16s model with alternative data splitting strategies:
# 1. **Within-Subject Split**: Train on _1 recordings, test on _2 recordings
# 2. **Leave-One-Subject-Out CV**: 10-fold cross-validation

# %% [markdown]
# ## 📦 Setup and Imports

# %%
import os
import sys
import keras
from keras.layers import concatenate
from sklearn.metrics import cohen_kappa_score
import math
import random
from keras import optimizers
import numpy as np
import scipy.io as spio
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
np.random.seed(0)

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

print("="*80)
print("🔬 Alternative Splits Training Pipeline (v2)")
print("="*80)

# %% [markdown]
# ## 🔧 Configuration

# %%
# ========================================
# CONFIGURATION - MODIFY AS NEEDED
# ========================================

# Experiment to run
# Options: 'within_subject', 'cross_validation'
EXPERIMENT = 'within_subject'  # Start with this (faster: ~1 hour)

# Base paths
DRIVE_BASE = '/content/drive/MyDrive/drowsiness-detection_-eog-based'
LOCAL_BASE = '/content'

# Paths
DATA_DIR = f'{DRIVE_BASE}/Phase1_CNN/data/processed/files/'
PHASE1_SRC = f'{DRIVE_BASE}/Phase1_CNN/src'
SPLITS_DIR = f'{DRIVE_BASE}/Alternative_Splits'
RESULTS_DIR = f'{DRIVE_BASE}/alternative_splits_results'

# Training parameters
BATCH_SIZE = 800
EPOCHS = 5
STRIDE = 1  # prec parameter

# Model parameters (CNN_16s)
FS = 200  # sampling rate
W_LEN = 8 * FS  # half window: 8 seconds * 200 Hz = 1600 samples (total = 16s)
DATA_DIM = W_LEN * 2  # full window: 3200 samples
N_CL = 2  # binary classification
N_CHANNELS = 2  # 2 EOG channels

print(f"✓ Configuration loaded")
print(f"  Experiment: {EXPERIMENT}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Stride: {STRIDE}")
print(f"  Window: {DATA_DIM} samples (16 seconds)")

# %% [markdown]
# ## 📁 Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print("✓ Google Drive mounted")

# %% [markdown]
# ## 📂 Verify Files

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
    print(f"\n❌ Data directory not found: {DATA_DIR}")
    raise FileNotFoundError(f"Data directory not found")

# Check Phase1_CNN source
if os.path.exists(PHASE1_SRC):
    print(f"✓ Phase1_CNN source found: {PHASE1_SRC}")
else:
    print(f"❌ Phase1_CNN source not found: {PHASE1_SRC}")
    raise FileNotFoundError(f"Phase1_CNN source not found")

# Check split files
if EXPERIMENT == 'within_subject':
    split_file = f'{SPLITS_DIR}/within_subject/file_sets.mat'
    if os.path.exists(split_file):
        print(f"✓ Within-Subject split found")
    else:
        print(f"❌ Split file not found: {split_file}")
        raise FileNotFoundError(f"Split file not found")
else:
    cv_dir = f'{SPLITS_DIR}/cross_validation'
    if os.path.exists(cv_dir):
        fold_files = [f for f in os.listdir(cv_dir) if f.startswith('fold_') and f.endswith('.mat')]
        print(f"✓ Cross-validation folds found: {len(fold_files)} folds")
    else:
        print(f"❌ CV directory not found: {cv_dir}")
        raise FileNotFoundError(f"CV directory not found")

print("\n" + "="*80)
print("✓ All files verified")
print("="*80)

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
# ## ⚙️ Import Phase1_CNN Modules

# %%
# Add Phase1_CNN to path
sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, f'{PHASE1_SRC}/models/cnn/CNN_16s')

# Import modules
from loadData import load_recording, classes_global, load_data
from utils import batch_generator, kappa_metric, create_tmp_dirs
from myModel import build_model

print("✓ Successfully imported Phase1_CNN modules")

# %% [markdown]
# ## 🔬 Experiment 1: Within-Subject Split

# %%
if EXPERIMENT == 'within_subject':
    print("="*80)
    print("Within-Subject Split Training")
    print("="*80)
    print("\nTrain: All _1 recordings (10 files)")
    print("Test: All _2 recordings (10 files)")
    print("\nThis tests temporal generalization within subjects.")
    print("="*80)
    
    # Load split
    split_file = f'{SPLITS_DIR}/within_subject/file_sets.mat'
    mat = spio.loadmat(split_file)
    
    files_train = []
    files_test = []
    
    # Parse train files
    tmp = mat['train_files'].flatten()
    for item in tmp:
        file = str(item[0]) + '.mat'
        files_train.append(file)
    
    # Parse test files
    tmp = mat['test_files'].flatten()
    for item in tmp:
        file = str(item[0]) + '.mat'
        files_test.append(file)
    
    print(f"\nTrain files ({len(files_train)}):")
    for f in files_train:
        print(f"  - {f}")
    
    print(f"\nTest files ({len(files_test)}):")
    for f in files_test:
        print(f"  - {f}")
    
    # Calculate class weights
    print("\n" + "="*80)
    print("Calculating class weights...")
    print("="*80)
    st0 = classes_global(DATA_DIR, files_train)
    cls = np.arange(N_CL)
    cl_w = class_weight.compute_class_weight('balanced', classes=cls, y=st0)
    print(f"Class weights: {cl_w}")
    
    # Load training data
    print("\n" + "="*80)
    print("Loading training data...")
    print("="*80)
    (data_train, targets_train, N_samples) = load_data(DATA_DIR, files_train, W_LEN)
    print(f'N_samples: {N_samples}')
    N_batches = int(math.ceil((N_samples + 0.0) / BATCH_SIZE))
    print(f'N_batches: {N_batches}')
    
    # Load test data
    print("\n" + "="*80)
    print("Loading test data...")
    print("="*80)
    (data_test, targets_test, N_samples_test) = load_data(DATA_DIR, files_test, W_LEN)
    print(f'N_samples_test: {N_samples_test}')
    
    # Create sample lists
    sample_list_train = []
    for i in range(len(targets_train)):
        for j in range(len(targets_train[i][0])):
            mid = j * STRIDE
            mid += W_LEN
            wnd_begin = mid - W_LEN
            wnd_end = mid + W_LEN - 1
            sample_list_train.append([i, j, wnd_begin, wnd_end])
    
    sample_list_test = []
    for i in range(len(targets_test)):
        sample_list_test.append([])
        for j in range(len(targets_test[i][0])):
            mid = j * STRIDE
            mid += W_LEN
            wnd_begin = mid - W_LEN
            wnd_end = mid + W_LEN - 1
            sample_list_test[i].append([i, j, wnd_begin, wnd_end])
    
    # Define generator
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
    
    # Build model
    print("\n" + "="*80)
    print("Building model...")
    print("="*80)
    ordering = 'channels_last'
    keras.backend.set_image_data_format(ordering)
    
    [cnn_eeg, model] = build_model(DATA_DIM, N_CHANNELS, N_CL)
    Nadam = optimizers.Nadam()
    model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    # Create results directory
    results_dir = f'{RESULTS_DIR}/within_subject'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/models', exist_ok=True)
    
    # Training loop
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
        model.save(f'{results_dir}/models/model_ep{epoch}.h5')
        print(f"✓ Model saved: epoch {epoch}")
    
    # Save final model
    model.save(f'{results_dir}/best_model.h5')
    print("\n✓ Training complete!")
    
    # Evaluate on test set
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
    
    # Calculate metrics
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
    
    # Save results
    results = {
        'y_true': test_y_true,
        'y_pred': test_y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'kappa': kappa,
        'confusion_matrix': cm,
        'train_files': files_train,
        'test_files': files_test
    }
    
    spio.savemat(f'{results_dir}/test_results.mat', results)
    print(f"\n✓ Results saved to {results_dir}")
    
    # Summary
    print("\n" + "="*80)
    print("WITHIN-SUBJECT SPLIT - SUMMARY")
    print("="*80)
    print(f"\nModel: CNN_16s")
    print(f"Train: {len(files_train)} files (all _1 recordings)")
    print(f"Test:  {len(files_test)} files (all _2 recordings)")
    print(f"\nTest Set Performance:")
    print(f"  Cohen's Kappa: {kappa:.3f}")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  F1-Score: {f1:.3f}")
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

else:
    print("Skipping within-subject (EXPERIMENT != 'within_subject')")

# %% [markdown]
# ## 🔬 Experiment 2: Leave-One-Subject-Out Cross-Validation
# 
# **Warning:** This will take 5-15 hours!

# %%
if EXPERIMENT == 'cross_validation':
    print("="*80)
    print("Leave-One-Subject-Out Cross-Validation")
    print("="*80)
    print("\n10-Fold CV: Each subject serves as test once")
    print("="*80)
    
    all_subjects = ['01M', '02F', '03F', '04M', '05M', '06M', '07F', '08M', '09M', '10M']
    num_folds = len(all_subjects)
    
    # Create results directory
    results_base_dir = f'{RESULTS_DIR}/cross_validation'
    os.makedirs(results_base_dir, exist_ok=True)
    
    all_results = []
    
    for fold in range(1, num_folds + 1):
        print("\n" + "="*80)
        print(f"FOLD {fold}/{num_folds} - Test Subject: {all_subjects[fold-1]}")
        print("="*80)
        
        # Load fold split
        fold_file = f'{SPLITS_DIR}/cross_validation/fold_{fold:02d}.mat'
        mat = spio.loadmat(fold_file)
        
        files_train = []
        files_test = []
        
        # Parse files
        tmp = mat['train_files'].flatten()
        for item in tmp:
            file = str(item[0]) + '.mat'
            files_train.append(file)
        
        tmp = mat['test_files'].flatten()
        for item in tmp:
            file = str(item[0]) + '.mat'
            files_test.append(file)
        
        print(f"\nTrain: {len(files_train)} files")
        print(f"Test:  {len(files_test)} files")
        
        # Calculate class weights
        st0 = classes_global(DATA_DIR, files_train)
        cls = np.arange(N_CL)
        cl_w = class_weight.compute_class_weight('balanced', classes=cls, y=st0)
        
        # Load data
        (data_train, targets_train, N_samples) = load_data(DATA_DIR, files_train, W_LEN)
        N_batches = int(math.ceil((N_samples + 0.0) / BATCH_SIZE))
        (data_test, targets_test, N_samples_test) = load_data(DATA_DIR, files_test, W_LEN)
        
        # Create sample lists
        sample_list_train = []
        for i in range(len(targets_train)):
            for j in range(len(targets_train[i][0])):
                mid = j * STRIDE
                mid += W_LEN
                wnd_begin = mid - W_LEN
                wnd_end = mid + W_LEN - 1
                sample_list_train.append([i, j, wnd_begin, wnd_end])
        
        sample_list_test = []
        for i in range(len(targets_test)):
            sample_list_test.append([])
            for j in range(len(targets_test[i][0])):
                mid = j * STRIDE
                mid += W_LEN
                wnd_begin = mid - W_LEN
                wnd_end = mid + W_LEN - 1
                sample_list_test[i].append([i, j, wnd_begin, wnd_end])
        
        # Define generator (same as within-subject)
        def my_generator(data, targets, sample_list, shuffle=True, class_weights=None):
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
                        sample_x1 = data[f][0][b:e+1]
                        sample_x2 = data[f][1][b:e+1]
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
                    batch_data = batch_data / 100
                    batch_data = np.clip(batch_data, -1, 1)
                    if class_weights is not None:
                        yield batch_data, batch_targets, batch_sample_weights
                    else:
                        yield batch_data, batch_targets
        
        # Build model
        print(f"\nBuilding model for fold {fold}...")
        ordering = 'channels_last'
        keras.backend.set_image_data_format(ordering)
        [cnn_eeg, model] = build_model(DATA_DIM, N_CHANNELS, N_CL)
        model.compile(optimizer='Nadam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Create fold results directory
        fold_results_dir = os.path.join(results_base_dir, f'fold_{fold:02d}')
        os.makedirs(fold_results_dir, exist_ok=True)
        os.makedirs(f'{fold_results_dir}/models', exist_ok=True)
        
        # Training
        print(f"\nTraining fold {fold}...")
        history = History()
        
        for epoch in range(EPOCHS):
            generator_train = my_generator(data_train, targets_train, sample_list_train, class_weights=cl_w)
            model.fit(generator_train, steps_per_epoch=N_batches, epochs=1, verbose=1, callbacks=[history])
            model.save(f'{fold_results_dir}/models/model_ep{epoch}.h5')
        
        model.save(f'{fold_results_dir}/best_model.h5')
        
        # Evaluate
        print(f"\nEvaluating fold {fold}...")
        test_y_pred = []
        test_y_true = []
        
        for j in range(len(data_test)):
            generator_test = my_generator(data_test, targets_test, sample_list_test[j], shuffle=False)
            y_pred = model.predict(generator_test, steps=int(math.ceil((len(sample_list_test[j]) + 0.0) / BATCH_SIZE)))
            test_y_pred.extend(np.argmax(y_pred, axis=1).flatten())
            test_y_true.extend(targets_test[j][0])
        
        test_y_pred = np.array(test_y_pred)
        test_y_true = np.array(test_y_true)
        
        # Calculate metrics
        accuracy = accuracy_score(test_y_true, test_y_pred)
        precision = precision_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
        recall = recall_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(test_y_true, test_y_pred, pos_label=1, zero_division=0)
        kappa = cohen_kappa_score(test_y_true, test_y_pred)
        cm = confusion_matrix(test_y_true, test_y_pred)
        
        # Store results
        fold_results = {
            'fold': fold,
            'test_subject': all_subjects[fold-1],
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'kappa': kappa,
            'confusion_matrix': cm,
            'y_true': test_y_true,
            'y_pred': test_y_pred
        }
        all_results.append(fold_results)
        
        # Save fold results
        spio.savemat(f'{fold_results_dir}/test_results.mat', fold_results)
        
        # Print fold results
        print(f"\nFold {fold} Results:")
        print(f"  Test Subject: {all_subjects[fold-1]}")
        print(f"  Kappa: {kappa:.3f}")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  Recall: {recall*100:.1f}%")
        print(f"  F1-Score: {f1:.3f}")
        print(f"\n✓ Fold {fold} complete!")
    
    # Aggregate CV results
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    kappas = [r['kappa'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    f1_scores = [r['f1_score'] for r in all_results]
    
    print(f"\nAggregated Results (Mean ± Std):")
    print(f"  Cohen's Kappa: {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
    print(f"  Precision: {np.mean(precisions)*100:.1f}% ± {np.std(precisions)*100:.1f}%")
    print(f"  Recall: {np.mean(recalls)*100:.1f}% ± {np.std(recalls)*100:.1f}%")
    print(f"  F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    
    print(f"\nPer-Fold Results:")
    for r in all_results:
        print(f"  Fold {r['fold']:02d} ({r['test_subject']}): Kappa={r['kappa']:.3f}, Precision={r['precision']*100:.1f}%, Recall={r['recall']*100:.1f}%")
    
    # Save aggregated results
    cv_summary = {
        'kappa_mean': np.mean(kappas),
        'kappa_std': np.std(kappas),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'all_kappas': kappas,
        'all_precisions': precisions,
        'all_recalls': recalls,
        'all_f1_scores': f1_scores,
        'test_subjects': [r['test_subject'] for r in all_results]
    }
    
    spio.savemat(os.path.join(results_base_dir, 'cv_summary.mat'), cv_summary)
    
    print(f"\n✓ CV summary saved to {results_base_dir}")
    print("\n" + "="*80)
    print("Cross-Validation Complete!")
    print("="*80)

else:
    print("Skipping cross-validation (EXPERIMENT != 'cross_validation')")

# %% [markdown]
# ## 💾 Results Summary

# %%
print("="*80)
print("Results saved to Google Drive")
print("="*80)
print(f"\nLocation: {RESULTS_DIR}")
print("\nYou can access them from your Google Drive!")

# %%

