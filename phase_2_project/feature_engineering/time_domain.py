"""
Time-Domain Feature Extraction for EOG Signals
==============================================

Extracts ~25 time-domain features including:
- Basic statistics: mean, std, var, min, max, range, median
- Shape features: peak/valley amplitude, rise/fall time
- Statistical moments: skewness, kurtosis
- Signal characteristics: zero-crossing rate, RMS, crest factor
- Hjorth parameters: activity, mobility, complexity

Author: Yaniv Grosberg
Date: 2025-11-09
"""

import numpy as np
from scipy.stats import skew, kurtosis


def extract_time_domain_features(signal):
    """
    Extract comprehensive time-domain features from an EOG signal segment.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array representing the EOG signal segment
        
    Returns
    -------
    features : dict
        Dictionary of feature_name: value pairs
    """
    features = {}
    
    # ============================================================
    # 1. Basic Statistics (7 features)
    # ============================================================
    features['td_mean'] = np.mean(signal)
    features['td_std'] = np.std(signal)
    features['td_var'] = np.var(signal)
    features['td_min'] = np.min(signal)
    features['td_max'] = np.max(signal)
    features['td_range'] = features['td_max'] - features['td_min']
    features['td_median'] = np.median(signal)
    
    # ============================================================
    # 2. Shape Features (8 features)
    # ============================================================
    features['td_peak_amp'] = np.max(signal)
    features['td_valley_amp'] = np.min(signal)
    
    # Peak and valley positions (normalized)
    peak_pos = np.argmax(signal)
    valley_pos = np.argmin(signal)
    features['td_peak_pos_norm'] = peak_pos / len(signal)
    features['td_valley_pos_norm'] = valley_pos / len(signal)
    
    # Rise and fall time (samples to reach peak/valley)
    signal_slope = np.gradient(signal)
    try:
        rise_time = np.where(signal >= features['td_peak_amp'])[0][0]
        features['td_rise_time'] = rise_time / len(signal)
    except:
        features['td_rise_time'] = 0.0
        
    try:
        fall_time = len(signal) - np.where(signal[::-1] >= features['td_peak_amp'])[0][0]
        features['td_fall_time'] = fall_time / len(signal)
    except:
        features['td_fall_time'] = 1.0
    
    # Inflection points (curvature changes)
    signal_curvature = np.gradient(signal_slope)
    inflection_points = np.where(np.diff(np.sign(signal_curvature)))[0]
    features['td_n_inflections'] = len(inflection_points)
    
    # Min to max ratio
    if features['td_peak_amp'] != 0:
        features['td_min_max_ratio'] = features['td_valley_amp'] / features['td_peak_amp']
    else:
        features['td_min_max_ratio'] = 0.0
    
    # ============================================================
    # 3. Statistical Moments (2 features)
    # ============================================================
    features['td_skewness'] = skew(signal)
    features['td_kurtosis'] = kurtosis(signal)
    
    # ============================================================
    # 4. Signal Characteristics (6 features)
    # ============================================================
    # Zero-crossing rate
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    features['td_zero_crossing_rate'] = len(zero_crossings) / len(signal)
    
    # RMS (Root Mean Square)
    features['td_rms'] = np.sqrt(np.mean(np.square(signal)))
    
    # Crest factor
    if features['td_rms'] != 0:
        features['td_crest_factor'] = features['td_peak_amp'] / features['td_rms']
    else:
        features['td_crest_factor'] = 0.0
    
    # Mean absolute deviation
    features['td_mean_abs_dev'] = np.mean(np.abs(signal - features['td_mean']))
    
    # Energy
    features['td_energy'] = np.sum(np.square(signal))
    
    # Area under curve
    features['td_auc'] = np.trapz(np.abs(signal))
    
    # ============================================================
    # 5. Hjorth Parameters (3 features)
    # ============================================================
    # Activity (variance)
    features['td_hjorth_activity'] = np.var(signal)
    
    # Mobility (standard deviation of derivative / standard deviation of signal)
    signal_diff = np.diff(signal)
    if features['td_std'] != 0:
        features['td_hjorth_mobility'] = np.std(signal_diff) / features['td_std']
    else:
        features['td_hjorth_mobility'] = 0.0
    
    # Complexity
    signal_diff2 = np.diff(signal_diff)
    if features['td_hjorth_mobility'] != 0 and np.std(signal_diff) != 0:
        mobility_of_derivative = np.std(signal_diff2) / np.std(signal_diff)
        features['td_hjorth_complexity'] = mobility_of_derivative / features['td_hjorth_mobility']
    else:
        features['td_hjorth_complexity'] = 0.0
    
    return features

