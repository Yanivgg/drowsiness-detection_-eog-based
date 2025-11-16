#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Display Detailed Results - Alternative Splits Experiments
==========================================================

This script reads saved results from Alternative_Splits experiments and displays
them in a detailed format with per-class metrics.

No training required - just loads and analyzes saved results!
"""

import os
import numpy as np
import scipy.io as spio
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    cohen_kappa_score
)
import argparse

# ============================================================================
# Configuration
# ============================================================================

# Update these paths to match your setup
# For Google Drive (Colab):
RESULTS_DIR_COLAB = '/content/drive/MyDrive/drowsiness-detection_-eog-based/alternative_splits_results'

# For local (Windows):
RESULTS_DIR_LOCAL = r'C:\Yaniv2\microsleep_drayad 2\drowsiness-detection_-eog-based\alternative_splits_results'

# Class names
CLASS_NAMES = ['Alert (0)', 'Drowsy (1)']

# ============================================================================
# Display Functions
# ============================================================================

def print_header(text, char='=', width=80):
    """Print a formatted header"""
    print()
    print(char * width)
    print(text.center(width))
    print(char * width)
    print()

def print_subheader(text, char='-', width=80):
    """Print a formatted subheader"""
    print()
    print(char * width)
    print(text)
    print(char * width)

def display_confusion_matrix(cm, class_names):
    """Display confusion matrix in a nice format"""
    print("\nConfusion Matrix:")
    print("-" * 50)
    
    # Header
    print(f"{'':>20} | {'Predicted':^30} |")
    print(f"{'':>20} | {class_names[0]:^13} | {class_names[1]:^13} |")
    print("-" * 50)
    
    # Rows
    for i, actual_class in enumerate(class_names):
        if i == 0:
            print(f"{'Actual':>12} {actual_class:>7} | {cm[i,0]:>13} | {cm[i,1]:>13} |")
        else:
            print(f"{' ':>12} {actual_class:>7} | {cm[i,0]:>13} | {cm[i,1]:>13} |")
    
    print("-" * 50)
    
    # Calculate percentages
    print("\nConfusion Matrix (Normalized by True Label):")
    print("-" * 50)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print(f"{'':>20} | {'Predicted':^30} |")
    print(f"{'':>20} | {class_names[0]:^13} | {class_names[1]:^13} |")
    print("-" * 50)
    
    for i, actual_class in enumerate(class_names):
        if i == 0:
            print(f"{'Actual':>12} {actual_class:>7} | {cm_normalized[i,0]:>12.1%} | {cm_normalized[i,1]:>12.1%} |")
        else:
            print(f"{' ':>12} {actual_class:>7} | {cm_normalized[i,0]:>12.1%} | {cm_normalized[i,1]:>12.1%} |")
    
    print("-" * 50)

def display_per_class_metrics(y_true, y_pred, class_names):
    """Display detailed per-class metrics"""
    print_subheader("Per-Class Metrics")
    
    # Calculate metrics for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Display table
    print(f"\n{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 65)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:>12.4f} {recall[i]:>12.4f} {f1[i]:>12.4f} {support[i]:>12.0f}")
    
    print("-" * 65)
    
    # Weighted averages
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)
    
    print(f"{'Weighted Avg':<15} {precision_weighted:>12.4f} {recall_weighted:>12.4f} {f1_weighted:>12.4f} {sum(support):>12.0f}")
    
    # Macro averages
    print(f"{'Macro Avg':<15} {np.mean(precision):>12.4f} {np.mean(recall):>12.4f} {np.mean(f1):>12.4f} {sum(support):>12.0f}")
    print()
    
    return precision, recall, f1, support

def display_overall_metrics(y_true, y_pred):
    """Display overall metrics"""
    print_subheader("Overall Metrics")
    
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"Accuracy:      {accuracy:>8.4f} ({accuracy*100:>6.2f}%)")
    print(f"Cohen's Kappa: {kappa:>8.4f}")
    print()

def display_class_distribution(y_true, class_names):
    """Display class distribution in dataset"""
    print_subheader("Class Distribution")
    
    unique, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    
    print(f"\n{'Class':<15} {'Count':>12} {'Percentage':>15}")
    print("-" * 45)
    
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"{class_names[cls]:<15} {count:>12} {percentage:>14.2f}%")
    
    print("-" * 45)
    print(f"{'Total':<15} {total:>12} {100.0:>14.2f}%")
    print()

def display_sklearn_report(y_true, y_pred, class_names):
    """Display sklearn's classification report"""
    print_subheader("Classification Report (sklearn)")
    
    # Get target names without numbers for cleaner output
    target_names = ['Alert', 'Drowsy']
    
    report = classification_report(
        y_true, y_pred, 
        target_names=target_names,
        zero_division=0,
        digits=4
    )
    print(report)

# ============================================================================
# Main Display Functions
# ============================================================================

def display_within_subject_results(results_dir):
    """Display Within-Subject Split results"""
    print_header("WITHIN-SUBJECT SPLIT - DETAILED RESULTS")
    
    results_file = os.path.join(results_dir, 'within_subject', 'test_results.mat')
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("\nPlease make sure you have:")
        print("1. Run the training script")
        print("2. Downloaded results from Google Drive (if using Colab)")
        return False
    
    # Load results
    print(f"📂 Loading results from: {results_file}")
    results = spio.loadmat(results_file)
    
    y_true = results['y_true'].flatten()
    y_pred = results['y_pred'].flatten()
    
    # Display experiment info
    print("\n📋 Experiment Information:")
    print(f"  Strategy: Within-Subject Split")
    print(f"  Train: All _1 recordings (10 files)")
    print(f"  Test:  All _2 recordings (10 files)")
    print(f"  Purpose: Test temporal generalization within subjects")
    
    # Display all metrics
    display_class_distribution(y_true, CLASS_NAMES)
    display_overall_metrics(y_true, y_pred)
    display_per_class_metrics(y_true, y_pred, CLASS_NAMES)
    display_confusion_matrix(results['confusion_matrix'], CLASS_NAMES)
    display_sklearn_report(y_true, y_pred, CLASS_NAMES)
    
    # Additional analysis
    print_subheader("Additional Analysis")
    
    # Sensitivity and Specificity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Specificity (True Negative Rate): {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"False Positive Rate:              {1-specificity:.4f} ({(1-specificity)*100:.2f}%)")
    print(f"False Negative Rate:              {1-sensitivity:.4f} ({(1-sensitivity)*100:.2f}%)")
    
    print()
    print("✅ Within-Subject results displayed successfully!")
    return True

def display_cross_validation_results(results_dir):
    """Display Cross-Validation results"""
    print_header("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION - DETAILED RESULTS")
    
    cv_dir = os.path.join(results_dir, 'cross_validation')
    
    if not os.path.exists(cv_dir):
        print(f"❌ CV directory not found: {cv_dir}")
        return False
    
    # Load summary file if exists
    summary_file = os.path.join(cv_dir, 'cv_summary.mat')
    if os.path.exists(summary_file):
        print(f"📂 Loading CV summary from: {summary_file}")
        summary = spio.loadmat(summary_file)
        
        print("\n📋 Cross-Validation Summary:")
        print(f"  Strategy: Leave-One-Subject-Out")
        print(f"  Number of folds: 10")
        print(f"  Purpose: Test generalization across subjects")
        
        print_subheader("Aggregated Results Across All Folds")
        
        print(f"Cohen's Kappa: {summary['kappa_mean'][0][0]:.4f} ± {summary['kappa_std'][0][0]:.4f}")
        print(f"Precision:     {summary['precision_mean'][0][0]:.4f} ± {summary['precision_std'][0][0]:.4f}")
        print(f"Recall:        {summary['recall_mean'][0][0]:.4f} ± {summary['recall_std'][0][0]:.4f}")
        print(f"F1-Score:      {summary['f1_mean'][0][0]:.4f} ± {summary['f1_std'][0][0]:.4f}")
    
    # Find all fold result files
    fold_files = []
    for i in range(1, 11):
        fold_dir = os.path.join(cv_dir, f'fold_{i:02d}')
        fold_file = os.path.join(fold_dir, 'test_results.mat')
        if os.path.exists(fold_file):
            fold_files.append((i, fold_file))
    
    if not fold_files:
        print("\n❌ No fold results found")
        return False
    
    print(f"\n📊 Found results for {len(fold_files)} folds")
    
    # Aggregate all predictions
    all_y_true = []
    all_y_pred = []
    
    # Display per-fold results
    print_subheader("Per-Fold Results")
    
    print(f"\n{'Fold':>6} {'Subject':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Kappa':>10}")
    print("-" * 80)
    
    for fold_num, fold_file in fold_files:
        results = spio.loadmat(fold_file)
        y_true = results['y_true'].flatten()
        y_pred = results['y_pred'].flatten()
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1, zero_division=0
        )
        kappa = cohen_kappa_score(y_true, y_pred)
        
        test_subject = results.get('test_subject', ['Unknown'])[0]
        if isinstance(test_subject, np.ndarray):
            test_subject = str(test_subject[0])
        else:
            test_subject = str(test_subject)
        
        print(f"{fold_num:>6} {test_subject:>10} {accuracy:>10.4f} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {kappa:>10.4f}")
    
    print("-" * 80)
    
    # Overall aggregated metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    print_header("AGGREGATED RESULTS - ALL FOLDS COMBINED")
    
    display_class_distribution(all_y_true, CLASS_NAMES)
    display_overall_metrics(all_y_true, all_y_pred)
    display_per_class_metrics(all_y_true, all_y_pred, CLASS_NAMES)
    
    cm_aggregated = confusion_matrix(all_y_true, all_y_pred)
    display_confusion_matrix(cm_aggregated, CLASS_NAMES)
    display_sklearn_report(all_y_true, all_y_pred, CLASS_NAMES)
    
    # Additional analysis
    print_subheader("Additional Analysis")
    
    tn, fp, fn, tp = cm_aggregated.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Specificity (True Negative Rate): {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"False Positive Rate:              {1-specificity:.4f} ({(1-specificity)*100:.2f}%)")
    print(f"False Negative Rate:              {1-sensitivity:.4f} ({(1-sensitivity)*100:.2f}%)")
    
    print()
    print("✅ Cross-Validation results displayed successfully!")
    return True

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Display detailed results from Alternative Splits experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display within-subject results (local path)
  python Display_Detailed_Results.py --experiment within_subject --local
  
  # Display CV results (Colab/Drive path)
  python Display_Detailed_Results.py --experiment cross_validation
  
  # Display both experiments
  python Display_Detailed_Results.py --experiment both --local
  
  # Use custom results directory
  python Display_Detailed_Results.py --results-dir /path/to/results --experiment within_subject
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['within_subject', 'cross_validation', 'both'],
        default='both',
        help='Which experiment results to display (default: both)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Custom results directory path'
    )
    
    parser.add_argument(
        '--local',
        action='store_true',
        help='Use local path instead of Colab/Drive path'
    )
    
    args = parser.parse_args()
    
    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    elif args.local:
        results_dir = RESULTS_DIR_LOCAL
    else:
        results_dir = RESULTS_DIR_COLAB
    
    print_header("📊 ALTERNATIVE SPLITS - DETAILED RESULTS VIEWER 📊")
    print(f"Results directory: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"\n❌ Results directory not found: {results_dir}")
        print("\nPlease:")
        print("1. Make sure you've run the training script")
        print("2. Update the paths at the top of this script")
        print("3. Or use --results-dir to specify the correct path")
        print("4. Or use --local flag if results are on local machine")
        return
    
    # Display requested experiments
    success = False
    
    if args.experiment in ['within_subject', 'both']:
        success = display_within_subject_results(results_dir) or success
    
    if args.experiment in ['cross_validation', 'both']:
        if args.experiment == 'both' and success:
            print("\n" * 2)
        success = display_cross_validation_results(results_dir) or success
    
    if success:
        print_header("✅ ALL RESULTS DISPLAYED SUCCESSFULLY! ✅")
    else:
        print_header("⚠️ NO RESULTS COULD BE DISPLAYED ⚠️")
        print("\nTroubleshooting:")
        print("1. Check that training has completed")
        print("2. Verify results directory path")
        print("3. If using Colab, download results from Google Drive")

if __name__ == '__main__':
    main()

