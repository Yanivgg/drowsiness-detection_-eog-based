# Results Analysis Notebook - Usage Guide

## Overview

The `results_analysis.ipynb` notebook provides comprehensive analysis and visualization of all trained microsleep detection models.

## Quick Start

### Option 1: Google Colab (Recommended)

1. **Upload to Colab**
   - Open Google Colab: https://colab.research.google.com
   - Upload `results_analysis.ipynb`

2. **Mount Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Verify Paths** (in Configuration cell)
   ```python
   BASE_PATH = '/content/drive/MyDrive/microsleep_cursor'
   RESULTS_PATH = '/content/drive/MyDrive/microsleep_results'
   ```

4. **Run All Cells**
   - Runtime → Run all
   - Or manually run each cell with Shift+Enter

### Option 2: Jupyter Notebook (Local)

1. **Install Dependencies**
   ```bash
   pip install numpy scipy matplotlib seaborn pandas scikit-learn jupyter
   ```

2. **Update Paths** (in Configuration cell)
   ```python
   BASE_PATH = './code'  # Path to code directory
   RESULTS_PATH = './results'  # Path to test results
   ```

3. **Launch Jupyter**
   ```bash
   jupyter notebook results_analysis.ipynb
   ```

4. **Run All Cells**

### Option 3: Python Script (No Jupyter)

1. **Run the Python version**
   ```bash
   python results_analysis.py
   ```

2. **Plots will be saved** to the OUTPUT_DIR specified in the script

## What You Need

### Required Files:

1. **Test Results** (from model predictions):
   - `CNN_2s_test_results.mat`
   - `CNN_4s_test_results.mat`
   - `CNN_8s_test_results.mat`
   - `CNN_16s_test_results.mat`
   - `CNN_32s_test_results.mat`
   - `CNN_LSTM_test_results.mat`

   Each `.mat` file should contain:
   - `y_true`: True labels
   - `y_pred`: Predicted labels
   - `accuracy`, `f1_score`, `kappa`, etc. (optional, will be recalculated)

2. **Training History** (optional, for training curves):
   - `code/CNN/CNN_*/predictions.mat`
   - `code/CNN_LSTM/predictions.mat`

   Each should contain:
   - `loss_tr`: Training loss per epoch
   - `loss_val`: Validation loss per epoch
   - `acc_val`: Validation kappa per epoch

## Outputs

The notebook generates:

### 1. Console Output
- Metrics summary table
- Window size analysis
- Key findings summary

### 2. Visualizations (saved to `OUTPUT_DIR`):
- `performance_comparison.png` - Bar charts for Kappa, Recall, Precision, Accuracy
- `precision_recall_tradeoff.png` - Scatter plot showing tradeoff
- `confusion_matrices.png` - Heatmaps for all models
- `training_curves.png` - Loss and Kappa vs epochs (if history available)
- `window_size_analysis.png` - Kappa vs window size
- `model_comparison.csv` - Comprehensive comparison table

### 3. Data Files:
- `model_comparison.csv` - All metrics in CSV format

## Customization

### Change Output Directory:
```python
OUTPUT_DIR = '/path/to/your/output'
```

### Analyze Subset of Models:
```python
MODELS = ['CNN_8s', 'CNN_16s', 'CNN_32s']  # Only these models
```

### Adjust Plot Style:
```python
plt.style.use('seaborn-v0_8-darkgrid')  # Change style
sns.set_palette("husl")  # Change color palette
plt.rcParams['figure.dpi'] = 100  # Change resolution
```

## Interpreting Results

### Cohen's Kappa (Primary Metric)
- **< 0.20**: Poor agreement
- **0.20-0.40**: Fair agreement  ← Most CNN models
- **0.40-0.60**: Moderate agreement  ← Target
- **0.60-0.80**: Substantial agreement
- **> 0.80**: Almost perfect agreement

### Precision vs Recall Tradeoff
- **High Precision, Low Recall**: Conservative model (few false alarms, misses real events)
- **Low Precision, High Recall**: Aggressive model (catches most events, many false alarms)
- **Balanced**: Best for practical applications

### Confusion Matrix
```
                 Predicted
                 Awake    Drowsy
Actual  Awake     TN        FP     ← Minimize FP (false alarms)
        Drowsy    FN        TP     ← Maximize TP (correct detections)
```

### Training Curves
- **Loss decreasing**: Model is learning
- **Validation loss increasing**: Overfitting
- **Kappa increasing**: Performance improving

### Window Size Analysis
- Shows optimal window size for this task
- Longer windows: More context, may lose temporal precision
- Shorter windows: Less context, may miss patterns

## Troubleshooting

### "File not found" errors
- Check that `BASE_PATH` and `RESULTS_PATH` are correct
- Verify test result files exist in `RESULTS_PATH`
- Make sure you've run predictions on test set first

### "No training history available"
- This is normal if you haven't saved training history
- Training curves will be skipped
- Other analyses will still work

### "Model X not found"
- The notebook handles missing models gracefully
- It will analyze only the models it finds
- Check that result files are named correctly: `{MODEL}_test_results.mat`

### Plots not displaying in Jupyter
- Try adding `%matplotlib inline` at the top
- Or use `%matplotlib notebook` for interactive plots

### Out of memory
- Reduce number of models analyzed
- Reduce plot DPI: `plt.rcParams['savefig.dpi'] = 150`

## Example Output

```
================================================================================
Metrics Summary (sorted by Kappa):
================================================================================
    Model  Accuracy  F1-Score  Precision    Recall  Specificity     Kappa
 CNN_16s    0.9680    0.9677     0.5494    0.3273       0.9904    0.3943
  CNN_8s    0.9680    0.9675     0.5891    0.2023       0.9952    0.2890
  CNN_4s    0.9619    0.9620     0.1972    0.0322       0.9990    0.0461
  CNN_2s    0.9434    0.9444     0.0946    0.0767       0.9971    0.0561
CNN_LSTM    0.9310    0.9336     0.0418    0.0455       0.9626    0.0080

================================================================================
Window Size Analysis:
================================================================================
Optimal Window Size: 16 seconds
Best Kappa: 0.3943
Best Model: CNN_16s
```

## Tips for Best Results

1. **Run all models first** before analysis
2. **Save test predictions** in the correct format
3. **Use consistent naming** for result files
4. **Mount Google Drive** if using Colab
5. **Check paths** before running
6. **Review all visualizations** - each tells a different story
7. **Save plots** for presentations/papers
8. **Export CSV** for further analysis in Excel/R

## Next Steps After Analysis

1. **Identify best model** (highest Kappa)
2. **Check precision-recall balance** for your use case
3. **Examine confusion matrix** of best model
4. **Review training curves** for signs of overfitting
5. **Consider retraining** best model with more epochs
6. **Test on additional subjects** if available
7. **Document findings** for your report/paper

## Questions?

Check the main README.md for more information about:
- Training models
- Data preprocessing
- Model architectures
- Performance metrics

