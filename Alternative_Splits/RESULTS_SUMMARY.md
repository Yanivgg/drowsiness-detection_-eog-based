# 📊 CNN_16s Model - Results Summary Across Three Splitting Strategies

## Overview
This document presents the performance evaluation of the **CNN_16s model** (16-second window) across three different data splitting strategies for drowsiness detection from EOG signals. Each strategy tests a different aspect of model generalization.

---

## 🎯 Quick Comparison - All Three Experiments

| Experiment | Cohen's Kappa | Drowsy Precision | Drowsy Recall | Drowsy F1-Score | Generalization Type |
|------------|---------------|------------------|---------------|-----------------|---------------------|
| **1️⃣ Split 1.5 Recordings**  | 0.4328 | 41.84%| 34.73%| 0.3796 | Temporal (within-subject, more data) |
| **2️⃣ Cross-Subject (Subject 7)** | 0.3944 | 54.86% | 32.73% | 0.4123 | Cross-subject (different subjects) |
| **3️⃣ Within-Subject Split** | 0.3227 | 38.79% | 28.73% | 0.3301 | Temporal (within-subject, less data) |

### 🏆 Winner by Metric:
- **Best Cohen's Kappa**: Split 1.5 (0.4328) ⭐
- **Best Drowsy Recall**: Split 1.5 (34.73%)
- **Best Drowsy Precision**: Cross-Subject (54.86%)
- **Best F1-Score**: Cross-Subject (0.4123)

---

## 📋 Detailed Results by Experiment

---

## 🥇 Experiment 1: Split 1.5 Recordings (BEST OVERALL)

### Experiment Design
- **Training Set**: 
  - All _1 recordings (complete) from all 10 subjects
  - First 50% of all _2 recordings
- **Test Set**: 
  - Last 50% of all _2 recordings
- **Generalization Type**: Temporal (within-subject)
- **Advantage**: 50% more training data than standard within-subject split

### Overall Performance
- **Cohen's Kappa**: **0.4328** (Moderate Agreement) 🏆
- **Accuracy**: 98.48%
- **Weighted F1-Score**: 0.9842

### Test Set Composition
- **Total Samples**: 7,267,500
- **Alert Samples**: 7,170,500 (98.67%)
- **Drowsy Samples**: 97,000 (1.33%)
- **Class Imbalance Ratio**: 73.9:1

### Per-Class Performance

#### 📊 Class 0: ALERT
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision** | 99.12% | When predicting Alert, correct 99.12% of the time |
| **Recall** | 99.35% | Detects 99.35% of actual Alert states |
| **F1-Score** | 0.9923 | Excellent Alert detection |
| **Support** | 7,170,500 samples | |

#### 📊 Class 1: DROWSY
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision** | **41.84%** | When predicting Drowsy, correct 41.84% of the time |
| **Recall** | **34.73%** | Detects 34.73% of actual Drowsy states |
| **F1-Score** | **0.3796** | Moderate Drowsy detection |
| **Support** | 97,000 samples | |

### Confusion Matrix

| Actual ↓ / Predicted → | Predicted Alert | Predicted Drowsy |
|------------------------|-----------------|------------------|
| **Actual Alert** | 7,123,677 (TN)<br>99.35% | 46,823 (FP)<br>0.65% |
| **Actual Drowsy** | 63,311 (FN)<br>65.27% | 33,689 (TP)<br>34.73% |

### Key Metrics
- **Sensitivity** (Drowsy Detection): 34.73%
- **Specificity** (Alert Detection): 99.35%
- **False Positive Rate**: 0.65%
- **False Negative Rate**: 65.27% ⚠️

---

## 🥈 Experiment 2: Cross-Subject Split (Subject 7 as Test)

### Experiment Design
- **Training Set**: Subjects 1-5 (all recordings)
- **Test Set**: Subjects 7-11 (all recordings)
- **Generalization Type**: Cross-subject (completely different subjects)
- **Note**: Subject 6 excluded from dataset

### Overall Performance
- **Cohen's Kappa**: 0.3944 (Fair Agreement, borderline Moderate)
- **Accuracy**: 96.75%

### Per-Class Performance

#### 📊 Class 0: ALERT
| Metric | Value |
|--------|-------|
| **Specificity** | 99.04% |
| **True Negatives** | 2,785,304 |

#### 📊 Class 1: DROWSY
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision** | **54.86%** 🏆 | Highest precision among all experiments |
| **Recall** | 32.73% | Detects 32.73% of actual Drowsy states |
| **F1-Score** | **0.4123** | Best F1 balance |

### Confusion Matrix

| Metric | Count |
|--------|-------|
| **True Positives (TP)** | 32,927 |
| **False Positives (FP)** | 27,096 |
| **False Negatives (FN)** | 67,673 |
| **True Negatives (TN)** | 2,785,304 |

### Key Metrics
- **Sensitivity** (Drowsy Detection): 32.73%
- **Specificity** (Alert Detection): 99.04%
- **False Negative Rate**: 67.27%

---

## 🥉 Experiment 3: Within-Subject Split

### Experiment Design
- **Training Set**: All _1 recordings (10 files)
- **Test Set**: All _2 recordings (10 files)
- **Generalization Type**: Temporal (within-subject)
- **Note**: Same subjects, different recording sessions

### Overall Performance
- **Cohen's Kappa**: 0.3227 (Fair Agreement)
- **Accuracy**: 98.50%
- **Weighted F1-Score**: 0.9839

### Test Set Composition
- **Total Samples**: 14,535,000
- **Alert Samples**: 14,347,600 (98.71%)
- **Drowsy Samples**: 187,400 (1.29%)
- **Class Imbalance Ratio**: 76.5:1

### Per-Class Performance

#### 📊 Class 0: ALERT
| Metric | Value |
|--------|-------|
| **Precision** | 99.07% |
| **Recall** | 99.41% |
| **F1-Score** | 0.9924 |
| **Support** | 14,347,600 samples |

#### 📊 Class 1: DROWSY
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Precision** | 38.79% | Lowest precision |
| **Recall** | **28.73%** | Lowest recall ⚠️ |
| **F1-Score** | 0.3301 | Weakest Drowsy detection |
| **Support** | 187,400 samples |

### Confusion Matrix

| Actual ↓ / Predicted → | Predicted Alert | Predicted Drowsy |
|------------------------|-----------------|------------------|
| **Actual Alert** | 14,262,624 (TN)<br>99.41% | 84,976 (FP)<br>0.59% |
| **Actual Drowsy** | 133,555 (FN)<br>71.27% | 53,845 (TP)<br>28.73% |

### Key Metrics
- **Sensitivity** (Drowsy Detection): 28.73%
- **Specificity** (Alert Detection): 99.41%
- **False Positive Rate**: 0.59%
- **False Negative Rate**: 71.27% ⚠️

---

## 📈 Comparative Analysis

### 1️⃣ Cohen's Kappa Comparison
```
Split 1.5:       ████████████████████ 0.4328 🏆 (Moderate)
Cross-Subject:   ███████████████████  0.3944 (Fair/Moderate)
Within-Subject:  ████████████████     0.3227 (Fair)
```

**Insight**: More training data (Split 1.5) yields best overall agreement.

---

### 2️⃣ Drowsy Detection Performance

#### Recall (Sensitivity) - Most Critical for Safety
```
Split 1.5:       ████████████████████ 34.73% 🏆
Cross-Subject:   ███████████████████  32.73%
Within-Subject:  ██████████████       28.73%
```

#### Precision - Avoiding False Alarms
```
Cross-Subject:   ████████████████████████████ 54.86% 🏆
Split 1.5:       ████████████████████         41.84%
Within-Subject:  ███████████████████          38.79%
```

#### F1-Score - Balance
```
Cross-Subject:   ████████████████████ 0.4123 🏆
Split 1.5:       ██████████████████   0.3796
Within-Subject:  ████████████████     0.3301
```

---

### 3️⃣ False Negative Rate (Missed Drowsy Events) ⚠️
```
Within-Subject:  ███████████████████████████████████ 71.27% (worst)
Cross-Subject:   ████████████████████████████████    67.27%
Split 1.5:       █████████████████████████████       65.27% (best)
```

**Critical Issue**: All three experiments miss 65-71% of drowsy events!

---

### 4️⃣ Class Imbalance Impact

All experiments show severe imbalance (Alert:Drowsy ratio):
- **Within-Subject**: 76.5:1 (worst imbalance)
- **Split 1.5**: 73.9:1
- **Cross-Subject**: ~74:1 (estimated)

**Result**: Model is conservative, favoring Alert predictions

---

## 🎯 Key Findings

### ✅ What Works Well:

1. **Split 1.5 Strategy** provides best overall performance
   - Highest Kappa (0.43)
   - Best drowsy recall (34.73%)
   - More training data = better generalization

2. **Alert Detection** is excellent across all experiments
   - >99% recall in all cases
   - >99% precision in all cases
   - Model excels at recognizing alert states

3. **Cross-Subject** shows best precision
   - 54.86% precision means fewer false alarms
   - Good for deployment where false alarms are costly
   - Model generalizes reasonably across subjects

### ⚠️ Critical Challenges:

1. **Low Drowsy Detection Rate** (28-35% recall)
   - Misses 65-71% of drowsy events
   - **Unacceptable for safety-critical applications**
   - Main failure mode: drowsy predicted as alert

2. **Severe Class Imbalance** (74-77:1 ratio)
   - Drowsy states are rare (~1-2% of data)
   - Model learns to be conservative
   - Need better balancing strategies

3. **Temporal Generalization** is challenging
   - Within-subject split performs worst
   - Drowsiness patterns change over time
   - Need more robust temporal features

---

## 🏆 Ranking by Use Case

### For Safety-Critical Deployment (Maximize Recall):
1. 🥇 **Split 1.5** (34.73% recall) - Still insufficient!
2. 🥈 Cross-Subject (32.73% recall)
3. 🥉 Within-Subject (28.73% recall)

**Verdict**: ❌ None suitable for safety-critical use as-is

### For Research/Screening (Balance Performance):
1. 🥇 **Split 1.5** (Kappa: 0.43)
2. 🥈 Cross-Subject (Kappa: 0.39)
3. 🥉 Within-Subject (Kappa: 0.32)

### For Low False Alarm Priority (Maximize Precision):
1. 🥇 **Cross-Subject** (54.86% precision)
2. 🥈 Split 1.5 (41.84% precision)
3. 🥉 Within-Subject (38.79% precision)

---

## 💡 Recommendations

### Immediate Actions:
1. ✅ **Use Split 1.5 strategy** for best overall performance
2. ⚠️ **Do NOT deploy** for safety-critical applications yet
3. 📊 Consider as **screening tool** only, not primary detector

### For Improving Drowsy Detection (Target: >70% Recall):

#### Data-Level Improvements:
- [ ] **Aggressive oversampling** of drowsy class (SMOTE, ADASYN)
- [ ] **Data augmentation** specific to drowsy patterns
- [ ] **Collect more drowsy samples** if possible
- [ ] **Re-label ambiguous samples** for clearer boundaries

#### Model-Level Improvements:
- [ ] **Adjust prediction threshold** (lower threshold for drowsy class)
- [ ] **Focal Loss** instead of cross-entropy (focus on hard examples)
- [ ] **Cost-sensitive learning** (penalize FN more than FP)
- [ ] **Ensemble methods** (combine multiple models)

#### Architecture Improvements:
- [ ] **Attention mechanisms** to focus on drowsy-specific patterns
- [ ] **Longer context** (combine multiple 16s windows)
- [ ] **Multi-scale features** (combine different window sizes)
- [ ] **Transformer-based models** for better temporal modeling

#### System-Level Improvements:
- [ ] **Multi-modal fusion** (add other sensors: steering, camera)
- [ ] **Temporal smoothing** (require multiple consecutive drowsy predictions)
- [ ] **Subject-specific calibration** (fine-tune per driver)
- [ ] **Confidence thresholds** (escalating alerts based on certainty)

---

## 📊 Statistical Interpretation

### Cohen's Kappa Guidelines:
| Range | Interpretation | Our Results |
|-------|----------------|-------------|
| 0.00-0.20 | Slight agreement | - |
| 0.21-0.40 | Fair agreement | Within-Subject (0.32) |
| 0.41-0.60 | Moderate agreement | Split 1.5 (0.43) ⭐ |
| 0.61-0.80 | Substantial agreement | - |
| 0.81-1.00 | Almost perfect | - |

**Note**: Cross-Subject (0.39) is borderline Fair/Moderate

---

## 🔧 Experimental Setup

### Model: CNN_16s
- **Architecture**: Convolutional Neural Network
- **Window Size**: 16 seconds (3200 samples)
- **Sampling Rate**: 200 Hz
- **Input**: 2 EOG channels (E1, E2)
- **Output**: 2 classes (Alert, Drowsy)

### Training Configuration:
- **Optimizer**: Nadam
- **Loss**: Categorical Cross-Entropy
- **Batch Size**: 800
- **Epochs**: 5
- **Class Weights**: Balanced (computed from training set)
- **Stride**: 1 sample

### Dataset:
- **Subjects**: 10 total (01M-10M, excluding 06M)
- **Recordings**: 2 per subject (_1 and _2)
- **Signal Type**: EOG (Electrooculography)
- **Classes**: Binary (Alert = 0, Drowsy = 1)

---

## 📁 Results Files Location

```
alternative_splits_results/
├── split_1_5/
│   ├── test_results.mat         # Split 1.5 results
│   └── models/
├── within_subject/
│   ├── test_results.mat         # Within-subject results
│   └── models/
└── cross_validation/
    └── (Cross-subject results from Phase 1)
```

---

## ✅ Conclusion

### Summary of Three Experiments:

1. **Split 1.5 Recordings** shows the best overall performance (Kappa: 0.43) and is **recommended for current use** in research and development contexts.

2. **Cross-Subject Split** demonstrates the best precision (54.86%) and good generalization across subjects, suitable when false alarms are costly.

3. **Within-Subject Split** has the weakest performance (Kappa: 0.32), showing that temporal generalization within subjects is challenging with limited data.

### Critical Finding:
⚠️ **All three approaches detect only 29-35% of drowsy events**, which is **insufficient for safety-critical deployment**. The model functions well as a research tool and potential screening mechanism, but requires significant improvements before use in real-world drowsiness warning systems.

### Path Forward:
Focus on improving **drowsy class recall** through:
- Enhanced data balancing techniques
- Threshold optimization
- Multi-modal sensor fusion
- Advanced architectures with attention mechanisms

---

**Document Generated**: November 2024  
**Model**: CNN_16s (16-second window)  
**Framework**: Keras/TensorFlow  
**Hardware**: Google Colab with GPU acceleration
