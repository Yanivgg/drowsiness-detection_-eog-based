# %% [markdown]
# # Phase 2: ML Training for Drowsiness Detection
# 
# **Train Random Forest & SVM on 16s EOG features**
# 
# - Train: Subjects 01-06, 08-10
# - Test: Subject 07 (same as CNN Phase 1)
# - Window: 16 seconds, Stride: 8 seconds (50% overlap)
# - Features: 63 features (time, frequency, non-linear, EOG-specific)

# %% [markdown]
# ## 1. Setup & Configuration

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import scipy.io as spio
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 2: ML Training for Drowsiness Detection")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %% [markdown]
# ## 2. Mount Google Drive (for saving models)

# %%
from google.colab import drive
drive.mount('/content/drive')

# Create results directory
RESULTS_DIR = '/content/drive/MyDrive/drowsiness_ml_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"\n[OK] Results will be saved to: {RESULTS_DIR}")

# %% [markdown]
# ## 3. Load Feature Data

# %%
print("\n" + "=" * 70)
print("Loading Feature Data")
print("=" * 70)

# Load train and test features
train_df = pd.read_csv('/content/train_features_16s.csv')
test_df = pd.read_csv('/content/test_features_16s.csv')

print(f"\n[OK] Train data: {len(train_df):,} windows")
print(f"[OK] Test data: {len(test_df):,} windows")

# Separate metadata from features
METADATA_COLS = ['Subject', 'Trial', 'Window', 'StartSample', 'EndSample', 'Label']
FEATURE_COLS = [col for col in train_df.columns if col not in METADATA_COLS]

print(f"\n[OK] Metadata columns: {len(METADATA_COLS)}")
print(f"[OK] Feature columns: {len(FEATURE_COLS)}")
print(f"\nFeature types breakdown:")
print(f"  - Time-domain: {len([f for f in FEATURE_COLS if f.startswith('td_')])}")
print(f"  - Frequency-domain: {len([f for f in FEATURE_COLS if f.startswith('fd_')])}")
print(f"  - Non-linear: {len([f for f in FEATURE_COLS if f.startswith('nl_')])}")
print(f"  - EOG-specific: {len([f for f in FEATURE_COLS if f.startswith('eog_')])}")

# Prepare train/test splits
X_train = train_df[FEATURE_COLS].values
y_train = train_df['Label'].values

X_test = test_df[FEATURE_COLS].values
y_test = test_df['Label'].values

print(f"\n[OK] X_train shape: {X_train.shape}")
print(f"[OK] X_test shape: {X_test.shape}")
print(f"[OK] y_train distribution: {np.bincount(y_train)} (0=awake, 1=drowsy)")
print(f"[OK] y_test distribution: {np.bincount(y_test)} (0=awake, 1=drowsy)")

# Calculate class distribution
train_drowsy_pct = 100 * np.sum(y_train == 1) / len(y_train)
test_drowsy_pct = 100 * np.sum(y_test == 1) / len(y_test)
print(f"\n📊 Train drowsy: {train_drowsy_pct:.2f}%")
print(f"📊 Test drowsy: {test_drowsy_pct:.2f}%")

# %% [markdown]
# ## 4. Train Random Forest Classifier

# %%
print("\n" + "=" * 70)
print("Training Random Forest Classifier")
print("=" * 70)

# Train with class balancing
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n[OK] Training Random Forest...")
rf_model.fit(X_train, y_train)
print("[OK] Training completed!")

# Evaluate on test set
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
rf_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'precision': precision_score(y_test, y_pred_rf),
    'recall': recall_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'kappa': cohen_kappa_score(y_test, y_pred_rf),
    'confusion_matrix': confusion_matrix(y_test, y_pred_rf)
}

print("\n" + "=" * 70)
print("Random Forest Results on Subject 07 Test Set")
print("=" * 70)
print(f"\n📊 Performance Metrics:")
print(f"  Accuracy:       {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.2f}%)")
print(f"  F1-score:       {rf_metrics['f1']:.4f}")
print(f"  Cohen's Kappa:  {rf_metrics['kappa']:.4f}")
print(f"\n🎯 Drowsy Detection (Class 1):")
print(f"  Precision: {rf_metrics['precision']:.4f} ({rf_metrics['precision']*100:.2f}%)")
print(f"  Recall:    {rf_metrics['recall']:.4f} ({rf_metrics['recall']*100:.2f}%)")
print(f"\n📉 Confusion Matrix:")
print("                 Predicted")
print("                 Awake    Drowsy")
cm = rf_metrics['confusion_matrix']
print(f"Actual  Awake   {cm[0,0]:,}  {cm[0,1]:,}")
print(f"        Drowsy  {cm[1,0]:,}  {cm[1,1]:,}")

# Save model
rf_model_path = os.path.join(RESULTS_DIR, 'random_forest_model.joblib')
joblib.dump(rf_model, rf_model_path)
print(f"\n✓ Model saved to: {rf_model_path}")

# %% [markdown]
# ## 5. Feature Importance Analysis

# %%
print("\n" + "=" * 70)
print("Feature Importance Analysis")
print("=" * 70)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Save feature importance
importance_path = os.path.join(RESULTS_DIR, 'feature_importance.csv')
feature_importance.to_csv(importance_path, index=False)
print(f"\n✓ Feature importance saved to: {importance_path}")

# Plot top 20 features
plt.figure(figsize=(12, 8))
top_20 = feature_importance.head(20)
plt.barh(range(len(top_20)), top_20['importance'])
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_plot.png'), dpi=150)
plt.show()
print("\n[OK] Feature importance plot saved")

# %% [markdown]
# ## 6. Train SVM Classifier

# %%
print("\n" + "=" * 70)
print("Training SVM Classifier")
print("=" * 70)

# Train SVM with class balancing
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    random_state=42,
    probability=True,  # Enable probability estimates
    verbose=True
)

print("\n[OK] Training SVM...")
svm_model.fit(X_train, y_train)
print("[OK] Training completed!")

# Evaluate on test set
y_pred_svm = svm_model.predict(X_test)
y_proba_svm = svm_model.predict_proba(X_test)[:, 1]

# Calculate metrics
svm_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'precision': precision_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm),
    'f1': f1_score(y_test, y_pred_svm),
    'kappa': cohen_kappa_score(y_test, y_pred_svm),
    'confusion_matrix': confusion_matrix(y_test, y_pred_svm)
}

print("\n" + "=" * 70)
print("SVM Results on Subject 07 Test Set")
print("=" * 70)
print(f"\n📊 Performance Metrics:")
print(f"  Accuracy:       {svm_metrics['accuracy']:.4f} ({svm_metrics['accuracy']*100:.2f}%)")
print(f"  F1-score:       {svm_metrics['f1']:.4f}")
print(f"  Cohen's Kappa:  {svm_metrics['kappa']:.4f}")
print(f"\n🎯 Drowsy Detection (Class 1):")
print(f"  Precision: {svm_metrics['precision']:.4f} ({svm_metrics['precision']*100:.2f}%)")
print(f"  Recall:    {svm_metrics['recall']:.4f} ({svm_metrics['recall']*100:.2f}%)")
print(f"\n📉 Confusion Matrix:")
print("                 Predicted")
print("                 Awake    Drowsy")
cm = svm_metrics['confusion_matrix']
print(f"Actual  Awake   {cm[0,0]:,}  {cm[0,1]:,}")
print(f"        Drowsy  {cm[1,0]:,}  {cm[1,1]:,}")

# Save model
svm_model_path = os.path.join(RESULTS_DIR, 'svm_model.joblib')
joblib.dump(svm_model, svm_model_path)
print(f"\n✓ Model saved to: {svm_model_path}")

# %% [markdown]
# ## 7. Model Comparison

# %%
print("\n" + "=" * 70)
print("Model Comparison on Subject 07 Test Set")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'SVM'],
    'Accuracy': [rf_metrics['accuracy'], svm_metrics['accuracy']],
    'Precision': [rf_metrics['precision'], svm_metrics['precision']],
    'Recall': [rf_metrics['recall'], svm_metrics['recall']],
    'F1-Score': [rf_metrics['f1'], svm_metrics['f1']],
    'Cohen\'s Kappa': [rf_metrics['kappa'], svm_metrics['kappa']]
})

print("\n" + comparison_df.to_string(index=False))

# Save comparison
comparison_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"\n✓ Comparison saved to: {comparison_path}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Metrics comparison
metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'Cohen\'s Kappa']
x = np.arange(len(metrics_to_plot))
width = 0.35

axes[0].bar(x - width/2, comparison_df[metrics_to_plot].iloc[0], width, label='Random Forest', alpha=0.8)
axes[0].bar(x + width/2, comparison_df[metrics_to_plot].iloc[1], width, label='SVM', alpha=0.8)
axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_to_plot, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim(0, 1)

# Confusion matrices side by side
cm_rf = rf_metrics['confusion_matrix']
cm_svm = svm_metrics['confusion_matrix']

# Normalize for better visualization
cm_rf_norm = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]
cm_svm_norm = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis]

im1 = axes[1].imshow(cm_rf_norm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title('Random Forest\nNormalized Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Awake', 'Drowsy'])
axes[1].set_yticklabels(['Awake', 'Drowsy'])

# Add text annotations
for i in range(2):
    for j in range(2):
        text = axes[1].text(j, i, f'{cm_rf_norm[i, j]:.2f}\n({cm_rf[i, j]:,})',
                           ha="center", va="center", color="white" if cm_rf_norm[i, j] > 0.5 else "black")

plt.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison_plot.png'), dpi=150)
plt.show()
print("\n[OK] Comparison plots saved")

# %% [markdown]
# ## 8. Save Complete Results

# %%
print("\n" + "=" * 70)
print("Saving Complete Results")
print("=" * 70)

# Save predictions
predictions_rf = pd.DataFrame({
    'Subject': test_df['Subject'],
    'Trial': test_df['Trial'],
    'Window': test_df['Window'],
    'True_Label': y_test,
    'RF_Prediction': y_pred_rf,
    'RF_Probability': y_proba_rf,
    'SVM_Prediction': y_pred_svm,
    'SVM_Probability': y_proba_svm
})
predictions_path = os.path.join(RESULTS_DIR, 'predictions_subject07.csv')
predictions_rf.to_csv(predictions_path, index=False)
print(f"[OK] Predictions saved to: {predictions_path}")

# Save all metrics to .mat file for consistency with Phase 1
results_mat = {
    'rf_metrics': {
        'accuracy': rf_metrics['accuracy'],
        'precision': rf_metrics['precision'],
        'recall': rf_metrics['recall'],
        'f1': rf_metrics['f1'],
        'kappa': rf_metrics['kappa'],
        'confusion_matrix': rf_metrics['confusion_matrix']
    },
    'svm_metrics': {
        'accuracy': svm_metrics['accuracy'],
        'precision': svm_metrics['precision'],
        'recall': svm_metrics['recall'],
        'f1': svm_metrics['f1'],
        'kappa': svm_metrics['kappa'],
        'confusion_matrix': svm_metrics['confusion_matrix']
    },
    'test_info': {
        'subject': '07F',
        'n_windows': len(y_test),
        'n_drowsy': int(np.sum(y_test == 1)),
        'n_awake': int(np.sum(y_test == 0))
    }
}
results_mat_path = os.path.join(RESULTS_DIR, 'ml_test_results.mat')
spio.savemat(results_mat_path, results_mat)
print(f"[OK] Results .mat file saved to: {results_mat_path}")

# Create summary report
summary_report = f"""
================================================================================
Phase 2: ML Training Results Summary
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFO:
- Train: Subjects 01-06, 08-10 ({len(X_train):,} windows, {train_drowsy_pct:.2f}% drowsy)
- Test: Subject 07 ({len(X_test):,} windows, {test_drowsy_pct:.2f}% drowsy)
- Window: 16 seconds (3200 samples)
- Stride: 8 seconds (50% overlap)
- Features: {len(FEATURE_COLS)} features

RANDOM FOREST RESULTS:
- Accuracy:  {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.2f}%)
- Precision: {rf_metrics['precision']:.4f} ({rf_metrics['precision']*100:.2f}%)
- Recall:    {rf_metrics['recall']:.4f} ({rf_metrics['recall']*100:.2f}%)
- F1-Score:  {rf_metrics['f1']:.4f}
- Kappa:     {rf_metrics['kappa']:.4f}

SVM RESULTS:
- Accuracy:  {svm_metrics['accuracy']:.4f} ({svm_metrics['accuracy']*100:.2f}%)
- Precision: {svm_metrics['precision']:.4f} ({svm_metrics['precision']*100:.2f}%)
- Recall:    {svm_metrics['recall']:.4f} ({svm_metrics['recall']*100:.2f}%)
- F1-Score:  {svm_metrics['f1']:.4f}
- Kappa:     {svm_metrics['kappa']:.4f}

FILES SAVED:
- Random Forest model: random_forest_model.joblib
- SVM model: svm_model.joblib
- Feature importance: feature_importance.csv
- Model comparison: model_comparison.csv
- Predictions: predictions_subject07.csv
- Results .mat: ml_test_results.mat
- Plots: feature_importance_plot.png, model_comparison_plot.png

================================================================================
"""

summary_path = os.path.join(RESULTS_DIR, 'training_summary.txt')
with open(summary_path, 'w') as f:
    f.write(summary_report)
print(summary_report)
print(f"✓ Summary report saved to: {summary_path}")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nAll results saved to: {RESULTS_DIR}")
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
