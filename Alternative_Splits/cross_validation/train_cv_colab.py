# %% [markdown]
# # Leave-One-Subject-Out Cross-Validation
# 
# Train 10 models, each with one subject as test and the other 9 as train.
# 
# **This script reuses all Phase1_CNN code** - only the data splits are different!

# %% Setup
import sys
import os
import numpy as np
import scipy.io as spio

# Add Phase1_CNN to path
sys.path.insert(0, '../../Phase1_CNN/src')
sys.path.insert(0, '../../Phase1_CNN')

print("="*80)
print("Leave-One-Subject-Out Cross-Validation")
print("="*80)
print("\n10-Fold CV: Each subject serves as test once")
print("="*80)

# %% Import Phase1_CNN modules
print("\nImporting Phase1_CNN modules...")

try:
    import loadData
    from models.cnn.CNN_16s import myModel
    import keras
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, cohen_kappa_score, confusion_matrix
    )
    print("✓ Successfully imported all modules")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    raise

# %% Configuration
DATA_DIR = '../../Phase1_CNN/data/processed/files'
RESULTS_BASE_DIR = './results/'
EPOCHS = 5
BATCH_SIZE = 800
STRIDE = 1

all_subjects = ['01M', '02F', '03F', '04M', '05M', '06M', '07F', '08M', '09M', '10M']
num_folds = len(all_subjects)

print(f"\nConfiguration:")
print(f"  Data dir: {DATA_DIR}")
print(f"  Results dir: {RESULTS_BASE_DIR}")
print(f"  Number of folds: {num_folds}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Stride: {STRIDE}")

# Create base results directory
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# %% Train all folds
all_results = []

for fold in range(1, num_folds + 1):
    print("\n" + "="*80)
    print(f"FOLD {fold}/{num_folds} - Test Subject: {all_subjects[fold-1]}")
    print("="*80)
    
    # Load fold split
    fold_file = f'./fold_{fold:02d}.mat'
    file_sets = spio.loadmat(fold_file)
    train_files = [f[0] for f in file_sets['train_files'][0]]
    test_files = [f[0] for f in file_sets['test_files'][0]]
    
    print(f"\nTrain: {len(train_files)} files (9 subjects)")
    print(f"Test:  {len(test_files)} files ({all_subjects[fold-1]})")
    
    # Create fold results directory
    fold_results_dir = os.path.join(RESULTS_BASE_DIR, f'fold_{fold:02d}')
    os.makedirs(fold_results_dir, exist_ok=True)
    
    # Create data generators
    print("\nCreating data generators...")
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
    
    print(f"✓ Train: {train_steps} steps/epoch")
    print(f"✓ Test: {test_steps} steps")
    
    # Build model
    print("\nBuilding model...")
    model = myModel.create_model()
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✓ Model compiled")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(fold_results_dir, 'best_model.h5'),
            monitor='loss',
            save_best_only=True,
            verbose=0
        ),
        EarlyStopping(
            monitor='loss',
            patience=3,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=2,
            verbose=0
        )
    ]
    
    # Train
    print(f"\nTraining fold {fold}...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Load best model and evaluate
    print(f"\nEvaluating fold {fold}...")
    model.load_weights(os.path.join(fold_results_dir, 'best_model.h5'))
    
    # Predict
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
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
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
        'y_true': y_true,
        'y_pred': y_pred
    }
    all_results.append(fold_results)
    
    # Save fold results
    spio.savemat(os.path.join(fold_results_dir, 'test_results.mat'), fold_results)
    
    # Print fold results
    print(f"\nFold {fold} Results:")
    print(f"  Test Subject: {all_subjects[fold-1]}")
    print(f"  Kappa: {kappa:.3f}")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  F1-Score: {f1:.3f}")
    print(f"\n✓ Fold {fold} complete!")

# %% Aggregate CV results
print("\n" + "="*80)
print("CROSS-VALIDATION SUMMARY")
print("="*80)

# Extract metrics
kappas = [r['kappa'] for r in all_results]
precisions = [r['precision'] for r in all_results]
recalls = [r['recall'] for r in all_results]
f1_scores = [r['f1_score'] for r in all_results]

# Calculate statistics
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

spio.savemat(os.path.join(RESULTS_BASE_DIR, 'cv_summary.mat'), cv_summary)

print(f"\n✓ CV summary saved to {RESULTS_BASE_DIR}")
print("\n" + "="*80)
print("Cross-Validation Complete!")
print("="*80)

# %%

