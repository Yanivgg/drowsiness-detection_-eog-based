"""
Frequency-Domain Feature Extraction for EOG Signals
===================================================

Extracts ~20 frequency-domain features including:
- Band powers: delta, theta, alpha, beta (absolute + relative)
- Spectral characteristics: centroid, spread, entropy, edge frequency
- Band power ratios: theta/alpha, delta/beta

Author: Yaniv Grosberg
Date: 2025-11-09
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy


def bandpower(signal, fs, band, nperseg=None):
    """
    Calculate band power using Welch's method.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal
    fs : float
        Sampling frequency
    band : tuple
        (low_freq, high_freq) in Hz
    nperseg : int, optional
        Length of each segment for Welch's method
        
    Returns
    -------
    power : float
        Band power
    """
    if nperseg is None:
        nperseg = min(len(signal), fs * 2)
    
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    freq_res = f[1] - f[0]
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx_band], dx=freq_res)


def extract_frequency_domain_features(signal, fs=200):
    """
    Extract comprehensive frequency-domain features from an EOG signal.
    
    Parameters
    ----------
    signal : np.ndarray
        1D array representing the EOG signal segment
    fs : float
        Sampling frequency in Hz (default: 200)
        
    Returns
    -------
    features : dict
        Dictionary of feature_name: value pairs
    """
    features = {}
    
    # ============================================================
    # 1. Frequency Bands (12 features: 4 absolute + 4 relative + 4 normalized)
    # ============================================================
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }
    
    # Calculate power spectrum
    nperseg = min(len(signal), fs * 2)
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    
    # Total power (0.5-30 Hz)
    total_power = bandpower(signal, fs, (0.5, 30), nperseg)
    
    # Extract band powers
    band_powers = {}
    for band_name, (low, high) in bands.items():
        power = bandpower(signal, fs, (low, high), nperseg)
        band_powers[band_name] = power
        
        # Absolute power
        features[f'fd_bp_{band_name}'] = power
        
        # Relative power (normalized by total power)
        if total_power > 0:
            features[f'fd_rel_{band_name}'] = power / total_power
        else:
            features[f'fd_rel_{band_name}'] = 0.0
    
    # ============================================================
    # 2. Band Power Ratios (4 features)
    # ============================================================
    # Theta/Alpha ratio (drowsiness indicator)
    if band_powers['alpha'] > 0:
        features['fd_theta_alpha_ratio'] = band_powers['theta'] / band_powers['alpha']
    else:
        features['fd_theta_alpha_ratio'] = 0.0
    
    # Delta/Beta ratio
    if band_powers['beta'] > 0:
        features['fd_delta_beta_ratio'] = band_powers['delta'] / band_powers['beta']
    else:
        features['fd_delta_beta_ratio'] = 0.0
    
    # (Delta+Theta)/(Alpha+Beta) ratio
    low_freq = band_powers['delta'] + band_powers['theta']
    high_freq = band_powers['alpha'] + band_powers['beta']
    if high_freq > 0:
        features['fd_low_high_ratio'] = low_freq / high_freq
    else:
        features['fd_low_high_ratio'] = 0.0
    
    # Alpha/Beta ratio
    if band_powers['beta'] > 0:
        features['fd_alpha_beta_ratio'] = band_powers['alpha'] / band_powers['beta']
    else:
        features['fd_alpha_beta_ratio'] = 0.0
    
    # ============================================================
    # 3. Spectral Characteristics (4 features)
    # ============================================================
    # Spectral centroid (center of mass of spectrum)
    if np.sum(Pxx) > 0:
        features['fd_spectral_centroid'] = np.sum(f * Pxx) / np.sum(Pxx)
    else:
        features['fd_spectral_centroid'] = 0.0
    
    # Spectral spread (standard deviation around centroid)
    if np.sum(Pxx) > 0:
        features['fd_spectral_spread'] = np.sqrt(
            np.sum(((f - features['fd_spectral_centroid']) ** 2) * Pxx) / np.sum(Pxx)
        )
    else:
        features['fd_spectral_spread'] = 0.0
    
    # Spectral entropy (measure of signal complexity)
    # Normalize power spectrum
    psd_norm = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else Pxx
    # Remove zeros for entropy calculation
    psd_norm = psd_norm[psd_norm > 0]
    features['fd_spectral_entropy'] = entropy(psd_norm)
    
    # Spectral edge frequency (95% of power)
    cumsum = np.cumsum(Pxx)
    if cumsum[-1] > 0:
        idx_edge = np.where(cumsum >= 0.95 * cumsum[-1])[0]
        if len(idx_edge) > 0:
            features['fd_spectral_edge'] = f[idx_edge[0]]
        else:
            features['fd_spectral_edge'] = f[-1]
    else:
        features['fd_spectral_edge'] = 0.0
    
    return features

