# %% [markdown]
# # Microsleep Detection - Results Analysis
# 
# Comprehensive analysis and comparison of all trained models:
# - CNN_2s, CNN_4s, CNN_8s, CNN_16s, CNN_32s
# - CNN_LSTM
# 
# **Test Set:** Subject 07 (100,600 microsleep samples)
# 
# **Key Metrics:**
# - Cohen's Kappa (primary metric for imbalanced data)
# - Precision & Recall
# - Confusion Matrix
# - Training Curves

# %% [markdown]
# ## 1. Setup and Configuration

# %%
# Import libraries
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, 
    precision_score, recall_score, cohen_kappa_score
)
import os
from pathlib import Path

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("Setup complete!")

# %% [markdown]
# ## 2. Configuration

# %%
# ========================================
# CONFIGURATION
# ========================================

# Base paths
BASE_PATH = '/content/drive/MyDrive/microsleep_cursor'  # Local files
RESULTS_PATH = '/content/drive/MyDrive/microsleep_results'  # Test results

# Model list
MODELS = ['CNN_2s', 'CNN_4s', 'CNN_8s', 'CNN_16s', 'CNN_32s', 'CNN_LSTM']

# Output directory for plots
OUTPUT_DIR = '/content/drive/MyDrive/microsleep_results/analysis_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Results path: {RESULTS_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Models to analyze: {len(MODELS)}")

# %% [markdown]
# ## 3. Load Test Results

# %%
print("="*80)
print("Loading Test Results from Google Drive")
print("="*80)

test_results = {}

for model_name in MODELS:
    result_file = f'{RESULTS_PATH}/{model_name}_test_results.mat'
    
    if os.path.exists(result_file):
        try:
            data = spio.loadmat(result_file)
            test_results[model_name] = {
                'y_true': data['y_true'].flatten(),
                'y_pred': data['y_pred'].flatten(),
                'accuracy': float(data['accuracy']) if 'accuracy' in data else None,
                'f1_score': float(data['f1_score']) if 'f1_score' in data else None,
                'kappa': float(data['kappa']) if 'kappa' in data else None,
                'recall_drowsy': float(data['recall_drowsy']) if 'recall_drowsy' in data else None,
                'precision_drowsy': float(data['precision_drowsy']) if 'precision_drowsy' in data else None,
                'confusion_matrix': data['confusion_matrix'] if 'confusion_matrix' in data else None
            }
            print(f"[OK] Loaded: {model_name}")
        except Exception as e:
            print(f"[X] Error loading {model_name}: {e}")
    else:
        print(f"[X] Not found: {model_name}")

print(f"\n[OK] Successfully loaded {len(test_results)}/{len(MODELS)} models")

# %% [markdown]
# ## 4. Load Training History

# %%
print("="*80)
print("Loading Training History")
print("="*80)

training_history = {}

for model_name in MODELS:
    if model_name == 'CNN_LSTM':
        history_file = f'{BASE_PATH}/code/CNN_LSTM/predictions.mat'
    else:
        history_file = f'{BASE_PATH}/code/CNN/{model_name}/predictions.mat'
    
    if os.path.exists(history_file):
        try:
            data = spio.loadmat(history_file)
            training_history[model_name] = {
                'loss_tr': data['loss_tr'].flatten() if 'loss_tr' in data else None,
                'loss_val': data['loss_val'].flatten() if 'loss_val' in data else None,
                'acc_val': data['acc_val'].flatten() if 'acc_val' in data else None,
                'K_val': data['K_val'] if 'K_val' in data else None
            }
            print(f"[OK] Loaded history: {model_name} ({len(training_history[model_name]['loss_tr'])} epochs)")
        except Exception as e:
            print(f"[X] Error loading history for {model_name}: {e}")
    else:
        print(f"[X] History not found: {model_name}")

print(f"\n[OK] Successfully loaded history for {len(training_history)}/{len(MODELS)} models")

# %% [markdown]
# ## 5. Calculate Performance Metrics

# %%
print("="*80)
print("Calculating Performance Metrics")
print("="*80)

metrics_summary = []

for model_name, data in test_results.items():
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics_summary.append({
        'Model': model_name,
        'Accuracy': acc,
        'F1-Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'Kappa': kappa,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    })
    
    print(f"\n{model_name}:")
    print(f"  Kappa:     {kappa:.4f}")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  TP: {tp:,}, FP: {fp:,}, FN: {fn:,}")

# Create DataFrame
df_metrics = pd.DataFrame(metrics_summary)
df_metrics = df_metrics.sort_values('Kappa', ascending=False)

print("\n" + "="*80)
print("Metrics Summary (sorted by Kappa):")
print("="*80)
print(df_metrics.to_string(index=False))

# %% [markdown]
# ## 6. Visualization: Performance Bar Charts

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Kappa comparison
ax1 = axes[0, 0]
df_sorted = df_metrics.sort_values('Kappa', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
bars1 = ax1.barh(df_sorted['Model'], df_sorted['Kappa'], color=colors)
ax1.set_xlabel("Cohen's Kappa", fontsize=12, fontweight='bold')
ax1.set_title("Model Comparison: Cohen's Kappa", fontsize=14, fontweight='bold')
ax1.set_xlim([0, max(df_sorted['Kappa']) * 1.15])
for i, v in enumerate(df_sorted['Kappa']):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Recall comparison
ax2 = axes[0, 1]
df_sorted = df_metrics.sort_values('Recall', ascending=True)
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df_sorted)))
bars2 = ax2.barh(df_sorted['Model'], df_sorted['Recall'], color=colors)
ax2.set_xlabel('Recall (Microsleep Detection)', fontsize=12, fontweight='bold')
ax2.set_title('Model Comparison: Recall', fontsize=14, fontweight='bold')
ax2.set_xlim([0, max(df_sorted['Recall']) * 1.15])
for i, v in enumerate(df_sorted['Recall']):
    ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Precision comparison
ax3 = axes[1, 0]
df_sorted = df_metrics.sort_values('Precision', ascending=True)
colors = plt.cm.cividis(np.linspace(0.3, 0.9, len(df_sorted)))
bars3 = ax3.barh(df_sorted['Model'], df_sorted['Precision'], color=colors)
ax3.set_xlabel('Precision (Microsleep Detection)', fontsize=12, fontweight='bold')
ax3.set_title('Model Comparison: Precision', fontsize=14, fontweight='bold')
ax3.set_xlim([0, max(df_sorted['Precision']) * 1.15])
for i, v in enumerate(df_sorted['Precision']):
    ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Accuracy comparison
ax4 = axes[1, 1]
df_sorted = df_metrics.sort_values('Accuracy', ascending=True)
colors = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(df_sorted)))
bars4 = ax4.barh(df_sorted['Model'], df_sorted['Accuracy'], color=colors)
ax4.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('Model Comparison: Accuracy', fontsize=14, fontweight='bold')
ax4.set_xlim([min(df_sorted['Accuracy']) * 0.98, 1.0])
for i, v in enumerate(df_sorted['Accuracy']):
    ax4.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[OK] Plot saved: {OUTPUT_DIR}/performance_comparison.png")

# %% [markdown]
# ## 7. Visualization: Precision-Recall Tradeoff

# %%
fig, ax = plt.subplots(figsize=(10, 8))

# Create scatter plot
colors = plt.cm.tab10(np.linspace(0, 1, len(df_metrics)))
for idx, (_, row) in enumerate(df_metrics.iterrows()):
    ax.scatter(row['Recall'], row['Precision'], s=300, alpha=0.7, 
              color=colors[idx], edgecolors='black', linewidth=2)
    ax.annotate(row['Model'], (row['Recall'], row['Precision']), 
               xytext=(8, 8), textcoords='offset points', 
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx], alpha=0.3))

ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=14, fontweight='bold')
ax.set_title('Precision-Recall Tradeoff\nMicrosleep Detection Performance', 
            fontsize=16, fontweight='bold')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, linestyle='--')

# Add diagonal reference line
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[OK] Plot saved: {OUTPUT_DIR}/precision_recall_tradeoff.png")

# %% [markdown]
# ## 8. Visualization: Confusion Matrices

# %%
n_models = len(test_results)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
axes = axes.flatten() if n_models > 1 else [axes]

for idx, (model_name, data) in enumerate(test_results.items()):
    y_true = data['y_true']
    y_pred = data['y_pred']
    
    cm = confusion_matrix(y_true, y_pred)
    
    ax = axes[idx]
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=',.0f', cmap='Blues', 
               ax=ax, cbar=True, square=True,
               annot_kws={'size': 12, 'weight': 'bold'})
    
    ax.set_title(f'{model_name}\nKappa: {df_metrics[df_metrics["Model"]==model_name]["Kappa"].values[0]:.3f}', 
                fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_xticklabels(['Awake', 'Drowsy'], fontsize=10)
    ax.set_yticklabels(['Awake', 'Drowsy'], fontsize=10, rotation=0)

# Hide unused subplots
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[OK] Plot saved: {OUTPUT_DIR}/confusion_matrices.png")

# %% [markdown]
# ## 9. Visualization: Training Curves

# %%
# Plot training curves for models with history
if len(training_history) > 0:
    n_models_with_history = len(training_history)
    n_cols = 2
    n_rows = n_models_with_history
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, history) in enumerate(training_history.items()):
        if history['loss_tr'] is not None:
            epochs = range(len(history['loss_tr']))
            
            # Plot 1: Loss curves
            ax1 = axes[idx, 0]
            ax1.plot(epochs, history['loss_tr'], 'o-', label='Training Loss', 
                    linewidth=2, markersize=8)
            if history['loss_val'] is not None:
                ax1.plot(epochs, history['loss_val'], 's-', label='Validation Loss', 
                        linewidth=2, markersize=8)
            ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax1.set_title(f'{model_name}: Training/Validation Loss', 
                         fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Validation Kappa
            ax2 = axes[idx, 1]
            if history['acc_val'] is not None:
                ax2.plot(epochs, history['acc_val'], 'o-', 
                        linewidth=2, markersize=8, color='green')
                ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                ax2.set_ylabel("Validation Kappa", fontsize=12, fontweight='bold')
                ax2.set_title(f'{model_name}: Validation Kappa', 
                             fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n[OK] Plot saved: {OUTPUT_DIR}/training_curves.png")
else:
    print("\n[WARN] No training history available")

# %% [markdown]
# ## 10. Window Size Analysis

# %%
# Extract CNN models and their window sizes
cnn_models = df_metrics[df_metrics['Model'].str.startswith('CNN_')].copy()
cnn_models = cnn_models[cnn_models['Model'] != 'CNN_LSTM']

# Extract window size from model name
cnn_models['Window_Size'] = cnn_models['Model'].str.extract(r'(\d+)').astype(int)
cnn_models = cnn_models.sort_values('Window_Size')

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot Kappa vs Window Size
ax.plot(cnn_models['Window_Size'], cnn_models['Kappa'], 
       'o-', linewidth=3, markersize=12, label='CNN Models', color='blue')

# Add CNN_LSTM as separate point
if 'CNN_LSTM' in df_metrics['Model'].values:
    lstm_kappa = df_metrics[df_metrics['Model']=='CNN_LSTM']['Kappa'].values[0]
    ax.scatter([35], [lstm_kappa], s=200, marker='*', 
              color='red', label='CNN_LSTM', zorder=10, edgecolors='black', linewidth=2)

# Annotate each point
for _, row in cnn_models.iterrows():
    ax.annotate(f"{row['Kappa']:.3f}", 
               (row['Window_Size'], row['Kappa']),
               xytext=(0, 10), textcoords='offset points',
               ha='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Window Size (seconds)', fontsize=14, fontweight='bold')
ax.set_ylabel("Cohen's Kappa", fontsize=14, fontweight='bold')
ax.set_title('Optimal Window Size Analysis\nKappa Performance vs Window Duration', 
            fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0, 40])
ax.set_ylim([0, max(cnn_models['Kappa']) * 1.15])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/window_size_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[OK] Plot saved: {OUTPUT_DIR}/window_size_analysis.png")

# Print analysis
print("\n" + "="*80)
print("Window Size Analysis:")
print("="*80)
best_window = cnn_models.loc[cnn_models['Kappa'].idxmax()]
print(f"\nOptimal Window Size: {int(best_window['Window_Size'])} seconds")
print(f"Best Kappa: {best_window['Kappa']:.4f}")
print(f"Best Model: {best_window['Model']}")

# %% [markdown]
# ## 11. Detailed Model Comparison Table

# %%
# Create comprehensive comparison table
comparison_data = []

for model_name in MODELS:
    row = {'Model': model_name}
    
    # Add test metrics
    if model_name in df_metrics['Model'].values:
        metrics = df_metrics[df_metrics['Model']==model_name].iloc[0]
        row['Kappa'] = f"{metrics['Kappa']:.4f}"
        row['Precision'] = f"{metrics['Precision']:.2%}"
        row['Recall'] = f"{metrics['Recall']:.2%}"
        row['Accuracy'] = f"{metrics['Accuracy']:.2%}"
        row['TP'] = f"{metrics['TP']:,}"
        row['FP'] = f"{metrics['FP']:,}"
        row['FN'] = f"{metrics['FN']:,}"
    
    # Add training info
    if model_name in training_history:
        history = training_history[model_name]
        if history['loss_tr'] is not None:
            row['Epochs'] = len(history['loss_tr'])
            if history['acc_val'] is not None:
                best_epoch = np.argmax(history['acc_val'])
                row['Best Epoch'] = best_epoch
                row['Best Val Kappa'] = f"{history['acc_val'][best_epoch]:.4f}"
    
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

print("="*80)
print("Comprehensive Model Comparison")
print("="*80)
print(df_comparison.to_string(index=False))

# Save to CSV
df_comparison.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)
print(f"\n[OK] Table saved: {OUTPUT_DIR}/model_comparison.csv")

# %% [markdown]
# ## 12. Key Findings Summary
# 
# ### Best Performing Model
# Based on Cohen's Kappa (the most appropriate metric for imbalanced data):
# 
# ### Performance Insights
# 1. **Optimal Window Size**: The analysis shows that window size significantly impacts performance
# 2. **Recall vs Precision Tradeoff**: Different models offer different balances
# 3. **Class Imbalance Challenge**: With only 3.45% drowsy samples, high accuracy doesn't indicate good performance
# 
# ### Model Rankings
# Models ranked by Cohen's Kappa (primary metric)

# %%
print("="*80)
print("KEY FINDINGS")
print("="*80)

# Best model
best_model = df_metrics.iloc[0]
print(f"\n[WIN] BEST MODEL: {best_model['Model']}")
print(f"   Kappa: {best_model['Kappa']:.4f}")
print(f"   Precision: {best_model['Precision']:.2%}")
print(f"   Recall: {best_model['Recall']:.2%}")
print(f"   True Positives: {best_model['TP']:,}")

# Worst model
worst_model = df_metrics.iloc[-1]
print(f"\n[X] WORST MODEL: {worst_model['Model']}")
print(f"   Kappa: {worst_model['Kappa']:.4f}")
print(f"   Precision: {worst_model['Precision']:.2%}")
print(f"   Recall: {worst_model['Recall']:.2%}")

# Optimal window
if 'Window_Size' in cnn_models.columns:
    optimal = cnn_models.loc[cnn_models['Kappa'].idxmax()]
    print(f"\n[TARGET] OPTIMAL WINDOW SIZE: {int(optimal['Window_Size'])} seconds")
    print(f"   Achieves best balance of context and precision")

# Performance summary
print(f"\n[STATS] OVERALL PERFORMANCE:")
print(f"   Average Kappa: {df_metrics['Kappa'].mean():.4f}")
print(f"   Kappa Range: {df_metrics['Kappa'].min():.4f} - {df_metrics['Kappa'].max():.4f}")
print(f"   Best Recall: {df_metrics['Recall'].max():.2%} ({df_metrics.loc[df_metrics['Recall'].idxmax(), 'Model']})")
print(f"   Best Precision: {df_metrics['Precision'].max():.2%} ({df_metrics.loc[df_metrics['Precision'].idxmax(), 'Model']})")

print("\n" + "="*80)
print("Analysis complete! All plots saved to:")
print(f"{OUTPUT_DIR}")
print("="*80)

