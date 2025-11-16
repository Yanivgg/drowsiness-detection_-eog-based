# %% [markdown]
# # Within-Subject Split Training
# 
# Train on _1 recordings, test on _2 recordings
# 
# **This script reuses all Phase1_CNN code** - only the data split is different!

# %% Setup
import sys
import os

# Add Phase1_CNN to path
sys.path.insert(0, '../../Phase1_CNN/src')
sys.path.insert(0, '../../Phase1_CNN')

print("="*80)
print("Within-Subject Split Training")
print("="*80)
print("\nTrain: All _1 recordings (10 files)")
print("Test: All _2 recordings (10 files)")
print("\nThis tests temporal generalization within subjects.")
print("="*80)

# %% Configuration
# Paths relative to this script
FILE_SETS_PATH = './file_sets.mat'  # Our custom split
DATA_DIR = '../../Phase1_CNN/data/processed/files'
RESULTS_DIR = './results/'
MODEL_NAME = 'CNN_16s_within_subject'

# Training parameters (same as Phase1)
EPOCHS = 5
BATCH_SIZE = 800
STRIDE = 1

print(f"\nConfiguration:")
print(f"  File sets: {FILE_SETS_PATH}")
print(f"  Data dir: {DATA_DIR}")
print(f"  Results dir: {RESULTS_DIR}")
print(f"  Model name: {MODEL_NAME}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Stride: {STRIDE}")

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# %% Import Phase1_CNN modules
print("\nImporting Phase1_CNN modules...")

try:
    import loadData
    from models.cnn.CNN_16s import myModel
    print("✓ Successfully imported Phase1_CNN modules")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    print("\nMake sure you're running this from Google Colab with the correct structure!")
    raise

# %% Load data using Phase1_CNN code
print("\n" + "="*80)
print("Loading Data")
print("="*80)

# Load file sets
import scipy.io as spio
file_sets = spio.loadmat(FILE_SETS_PATH)
train_files = [f[0] for f in file_sets['train_files'][0]]
test_files = [f[0] for f in file_sets['test_files'][0]]

print(f"\nTrain files: {len(train_files)}")
for f in train_files:
    print(f"  - {f}")

print(f"\nTest files: {len(test_files)}")
for f in test_files:
    print(f"  - {f}")

# %% Create data generators
print("\n" + "="*80)
print("Creating Data Generators")
print("="*80)

# Use Phase1_CNN's loadData module
train_gen, train_steps, class_weights = loadData.create_generator(
    train_files,
    DATA_DIR,
    batch_size=BATCH_SIZE,
    stride=STRIDE,
    shuffle=True
)

test_gen, test_steps, _ = loadData.create_generator(
    test_files,
    DATA_DIR,
    batch_size=BATCH_SIZE,
    stride=STRIDE,
    shuffle=False
)

print(f"\n✓ Train generator: {train_steps} steps/epoch")
print(f"✓ Test generator: {test_steps} steps")
print(f"✓ Class weights: {class_weights}")

# %% Build model
print("\n" + "="*80)
print("Building Model")
print("="*80)

# Use Phase1_CNN's model architecture
model = myModel.create_model()
model.summary()

# Compile with class weights
import keras
model.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model compiled")

# %% Train model
print("\n" + "="*80)
print("Training Model")
print("="*80)

# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint(
        os.path.join(RESULTS_DIR, 'best_model.h5'),
        monitor='loss',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='loss',
        patience=3,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        verbose=1
    )
]

# Train
history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

print("\n✓ Training complete!")

# %% Evaluate on test set
print("\n" + "="*80)
print("Evaluating on Test Set")
print("="*80)

# Load best model
model.load_weights(os.path.join(RESULTS_DIR, 'best_model.h5'))

# Predict
import numpy as np
y_pred = []
y_true = []

for i in range(test_steps):
    X_batch, y_batch = next(test_gen)
    pred_batch = model.predict(X_batch, verbose=0)
    y_pred.extend(np.argmax(pred_batch, axis=1))
    y_true.extend(y_batch)

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Calculate metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, cohen_kappa_score, confusion_matrix
)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
kappa = cohen_kappa_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

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
    'y_true': y_true,
    'y_pred': y_pred,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'kappa': kappa,
    'confusion_matrix': cm
}

spio.savemat(os.path.join(RESULTS_DIR, 'test_results.mat'), results)
print(f"\n✓ Results saved to {RESULTS_DIR}")

# %% Summary
print("\n" + "="*80)
print("WITHIN-SUBJECT SPLIT - SUMMARY")
print("="*80)
print(f"\nModel: CNN_16s")
print(f"Train: {len(train_files)} files (all _1 recordings)")
print(f"Test:  {len(test_files)} files (all _2 recordings)")
print(f"\nTest Set Performance:")
print(f"  Cohen's Kappa: {kappa:.3f}")
print(f"  Precision: {precision*100:.1f}%")
print(f"  Recall: {recall*100:.1f}%")
print(f"  F1-Score: {f1:.3f}")
print("\n" + "="*80)
print("Training Complete!")
print("="*80)

# %%

