"""
EOG-Specific Feature Extraction
================================

Extracts 12 EOG-specific features including:
- Blink detection and characteristics (3)
- LOC-ROC correlation and phase (3)
- Eye movement patterns (4)
- Asymmetry features (2)

Note: SEM (Slow Eye Movements) features removed - detection not working properly


Updated: 2025-11-11 (removed SEM features)
"""

import numpy as np
from scipy.signal import find_peaks, correlate
from scipy.stats import pearsonr


def detect_blinks(signal, fs=200, min_amplitude=None, min_distance=None):
    """
    Detect blinks in EOG signal.
    
    Blinks appear as sharp positive or negative deflections.
    
    Parameters
    ----------
    signal : np.ndarray
        EOG signal
    fs : float
        Sampling frequency
    min_amplitude : float, optional
        Minimum peak amplitude (default: 1.5 * std)
    min_distance : int, optional
        Minimum distance between peaks in samples (default: 0.2s)
        
    Returns
    -------
    blinks : np.ndarray
        Array of blink indices
    amplitudes : np.ndarray
        Array of blink amplitudes
    """
    if min_amplitude is None:
        min_amplitude = 1.5 * np.std(signal)
    if min_distance is None:
        min_distance = int(0.2 * fs)  # 200ms minimum between blinks
    
    # Find positive and negative peaks
    pos_peaks, pos_props = find_peaks(signal, height=min_amplitude, distance=min_distance)
    neg_peaks, neg_props = find_peaks(-signal, height=min_amplitude, distance=min_distance)
    
    # Combine peaks
    all_peaks = np.concatenate([pos_peaks, neg_peaks])
    all_amps = np.concatenate([pos_props['peak_heights'], neg_props['peak_heights']])
    
    # Sort by time
    sort_idx = np.argsort(all_peaks)
    
    return all_peaks[sort_idx], all_amps[sort_idx]


def detect_slow_eye_movements(signal, fs=200, threshold=0.05):
    """
    Detect Slow Eye Movements (SEMs) - characteristic of drowsiness.
    
    SEMs are slow, rolling eye movements with frequencies 0.1-0.5 Hz.
    
    Parameters
    ----------
    signal : np.ndarray
        EOG signal
    fs : float
        Sampling frequency
    threshold : float
        Velocity threshold for SEM detection
        
    Returns
    -------
    sem_count : int
        Number of detected SEMs
    sem_duration : float
        Total duration of SEMs (normalized)
    """
    # Calculate velocity (first derivative)
    velocity = np.abs(np.diff(signal))
    
    # SEMs are characterized by low velocity sustained movements
    sem_mask = velocity < threshold
    
    # Find continuous SEM regions
    sem_regions = []
    in_sem = False
    start = 0
    
    for i, is_sem in enumerate(sem_mask):
        if is_sem and not in_sem:
            start = i
            in_sem = True
        elif not is_sem and in_sem:
            if i - start > fs * 0.5:  # At least 0.5s duration
                sem_regions.append((start, i))
            in_sem = False
    
    sem_count = len(sem_regions)
    sem_duration = sum([end - start for start, end in sem_regions]) / len(signal)
    
    return sem_count, sem_duration


def extract_eog_specific_features(loc_signal, roc_signal, fs=200):
    """
    Extract EOG-specific features from LOC and ROC channels.
    
    Parameters
    ----------
    loc_signal : np.ndarray
        Left EOG (LOC) signal
    roc_signal : np.ndarray
        Right EOG (ROC) signal
    fs : float
        Sampling frequency (default: 200 Hz)
        
    Returns
    -------
    features : dict
        Dictionary of feature_name: value pairs
    """
    features = {}
    
    # ============================================================
    # 1. Blink Characteristics (3 features)
    # ============================================================
    # Detect blinks in LOC
    blinks_loc, amps_loc = detect_blinks(loc_signal, fs)
    features['eog_blink_rate'] = len(blinks_loc) / (len(loc_signal) / fs)  # blinks per second
    
    if len(amps_loc) > 0:
        features['eog_blink_amp_mean'] = np.mean(amps_loc)
        features['eog_blink_amp_std'] = np.std(amps_loc)
    else:
        features['eog_blink_amp_mean'] = 0.0
        features['eog_blink_amp_std'] = 0.0
    
    # ============================================================
    # 2. Slow Eye Movements - REMOVED (always returns 0)
    # ============================================================
    # sem_count, sem_duration = detect_slow_eye_movements(loc_signal, fs)
    # features['eog_sem_count'] = sem_count
    # features['eog_sem_duration'] = sem_duration
    # NOTE: SEM detection not working properly - threshold too sensitive
    
    # ============================================================
    # 3. LOC-ROC Correlation (3 features)
    # ============================================================
    # Pearson correlation (measures synchrony)
    try:
        corr, p_value = pearsonr(loc_signal, roc_signal)
        features['eog_loc_roc_corr'] = corr
    except:
        features['eog_loc_roc_corr'] = 0.0
    
    # Cross-correlation lag (measures phase difference)
    try:
        cross_corr = correlate(loc_signal - np.mean(loc_signal), 
                              roc_signal - np.mean(roc_signal), mode='same')
        lag = np.argmax(np.abs(cross_corr)) - len(cross_corr) // 2
        features['eog_loc_roc_lag'] = lag / fs  # lag in seconds
        features['eog_loc_roc_lag_corr'] = np.max(np.abs(cross_corr)) / len(loc_signal)
    except:
        features['eog_loc_roc_lag'] = 0.0
        features['eog_loc_roc_lag_corr'] = 0.0
    
    # ============================================================
    # 4. Differential Signal Characteristics (4 features)
    # ============================================================
    # Horizontal EOG (differential)
    diff_signal = loc_signal - roc_signal
    
    # Eye movement amplitude (range of differential)
    features['eog_movement_amplitude'] = np.max(diff_signal) - np.min(diff_signal)
    
    # Eye movement velocity (mean absolute derivative)
    velocity = np.abs(np.diff(diff_signal))
    features['eog_movement_velocity_mean'] = np.mean(velocity)
    features['eog_movement_velocity_std'] = np.std(velocity)
    
    # Saccade detection (rapid eye movements)
    # Saccades are characterized by high velocity peaks
    velocity_threshold = np.mean(velocity) + 2 * np.std(velocity)
    saccades, _ = find_peaks(velocity, height=velocity_threshold, distance=int(0.02 * fs))
    features['eog_saccade_rate'] = len(saccades) / (len(loc_signal) / fs)  # saccades per second
    
    # ============================================================
    # 5. Asymmetry Features (2 features)
    # ============================================================
    # Measure asymmetry between LOC and ROC
    features['eog_loc_roc_energy_ratio'] = (np.sum(loc_signal ** 2) / 
                                            (np.sum(roc_signal ** 2) + 1e-10))
    
    features['eog_loc_roc_amplitude_ratio'] = ((np.max(np.abs(loc_signal))) / 
                                               (np.max(np.abs(roc_signal)) + 1e-10))
    
    return features

