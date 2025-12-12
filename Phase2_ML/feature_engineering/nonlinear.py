"""
Non-Linear Feature Extraction for EOG Signals
=============================================

Extracts ~10 non-linear features including:
- Entropy measures: Sample Entropy, Approximate Entropy
- Fractal dimension: Higuchi Fractal Dimension
- Detrended Fluctuation Analysis (DFA)
- Lyapunov exponent (optional, computationally expensive)


"""

import numpy as np


def sample_entropy(signal, m=2, r=None):
    """
    Calculate Sample Entropy (SampEn) of a signal.
    
    Sample Entropy is a measure of complexity/regularity.
    Lower values indicate more self-similar/regular patterns.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal
    m : int
        Embedding dimension (default: 2)
    r : float, optional
        Tolerance (default: 0.2 * std of signal)
        
    Returns
    -------
    sampen : float
        Sample entropy value
    """
    N = len(signal)
    if r is None:
        r = 0.2 * np.std(signal)
    
    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        x = [[signal[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) 
             for i in range(len(x))]
        return sum(C)
    
    try:
        return -np.log(_phi(m + 1) / _phi(m))
    except:
        return 0.0


def approximate_entropy(signal, m=2, r=None):
    """
    Calculate Approximate Entropy (ApEn) of a signal.
    
    Similar to Sample Entropy but includes self-matches.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal
    m : int
        Embedding dimension (default: 2)
    r : float, optional
        Tolerance (default: 0.2 * std of signal)
        
    Returns
    -------
    apen : float
        Approximate entropy value
    """
    N = len(signal)
    if r is None:
        r = 0.2 * np.std(signal)
    
    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        x = [[signal[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) 
             for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    
    try:
        return abs(_phi(m) - _phi(m + 1))
    except:
        return 0.0


def higuchi_fractal_dimension(signal, kmax=10):
    """
    Calculate Higuchi Fractal Dimension (HFD).
    
    Measures the complexity and self-similarity of a time series.
    Higher values indicate more complex/irregular signals.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal
    kmax : int
        Maximum k value (default: 10)
        
    Returns
    -------
    hfd : float
        Higuchi fractal dimension
    """
    N = len(signal)
    L = []
    x = []
    
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            Lmk = 0
            for i in range(1, int((N - m) / k)):
                Lmk += abs(signal[m + i * k] - signal[m + (i - 1) * k])
            Lmk = Lmk * (N - 1) / (k * k * int((N - m) / k))
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
        x.append(np.log(1.0 / k))
    
    # Linear regression
    try:
        hfd = np.polyfit(x, L, 1)[0]
        return hfd
    except:
        return 1.0


def detrended_fluctuation_analysis(signal, min_box=4, max_box=None):
    """
    Calculate Detrended Fluctuation Analysis (DFA) exponent.
    
    DFA quantifies long-range correlations in non-stationary signals.
    α = 0.5: white noise, α = 1: 1/f noise, α = 1.5: Brownian noise
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal
    min_box : int
        Minimum box size (default: 4)
    max_box : int, optional
        Maximum box size (default: N/4)
        
    Returns
    -------
    alpha : float
        DFA scaling exponent
    """
    N = len(signal)
    if max_box is None:
        max_box = N // 4
    
    # Integrate the signal
    y = np.cumsum(signal - np.mean(signal))
    
    # Box sizes (logarithmically spaced)
    scales = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 20).astype(int))
    
    fluctuations = []
    for scale in scales:
        # Divide into boxes
        n_boxes = N // scale
        boxes = [y[i * scale:(i + 1) * scale] for i in range(n_boxes)]
        
        # Calculate fluctuation for each box
        F = []
        for box in boxes:
            if len(box) < 2:
                continue
            # Fit polynomial (order 1 = linear detrending)
            t = np.arange(len(box))
            coeffs = np.polyfit(t, box, 1)
            fit = np.polyval(coeffs, t)
            # Calculate fluctuation
            F.append(np.sqrt(np.mean((box - fit) ** 2)))
        
        if len(F) > 0:
            fluctuations.append(np.mean(F))
        else:
            fluctuations.append(0)
    
    # Remove zeros for log-log regression
    scales_valid = scales[np.array(fluctuations) > 0]
    fluctuations_valid = np.array(fluctuations)[np.array(fluctuations) > 0]
    
    if len(scales_valid) > 1:
        try:
            # Linear fit in log-log space
            log_scales = np.log10(scales_valid)
            log_fluct = np.log10(fluctuations_valid)
            alpha = np.polyfit(log_scales, log_fluct, 1)[0]
            return alpha
        except:
            return 1.0
    else:
        return 1.0


def extract_nonlinear_features(signal):
    """
    Extract FAST non-linear features from an EOG signal.
    
    NOTE: Sample/Approximate Entropy removed due to O(N²) complexity.
          With 3200 samples, they take 30-40 seconds per window!
    
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
    # REMOVED: Sample Entropy & Approximate Entropy (TOO SLOW!)
    # These are O(N²) algorithms - 39 seconds per window
    # ============================================================
    
    # ============================================================
    # 1. Fractal Dimension (1 feature) - FAST
    # ============================================================
    try:
        features['nl_higuchi_fd'] = higuchi_fractal_dimension(signal, kmax=10)
    except:
        features['nl_higuchi_fd'] = 1.0
    
    # ============================================================
    # 2. Detrended Fluctuation Analysis (1 feature) - MEDIUM SPEED
    # ============================================================
    # SKIP DFA as well - it's also slow with 3200 samples
    # try:
    #     features['nl_dfa_alpha'] = detrended_fluctuation_analysis(signal)
    # except:
    #     features['nl_dfa_alpha'] = 1.0
    
    # ============================================================
    # 4. Additional Complexity Measures (6 features)
    # ============================================================
    # Petrosian Fractal Dimension
    N = len(signal)
    diff = np.diff(signal)
    N_delta = len(np.where(diff[:-1] * diff[1:] < 0)[0])  # sign changes
    if N_delta > 0:
        features['nl_petrosian_fd'] = np.log10(N) / (
            np.log10(N) + np.log10(N / (N + 0.4 * N_delta))
        )
    else:
        features['nl_petrosian_fd'] = 1.0
    
    # Katz Fractal Dimension
    dists = np.abs(diff)
    L = np.sum(dists)
    d = np.max(np.abs(signal - signal[0]))
    if d > 0 and L > 0:
        features['nl_katz_fd'] = np.log10(L) / np.log10(d)
    else:
        features['nl_katz_fd'] = 1.0
    
    # Hurst exponent (R/S analysis - simplified)
    try:
        mean = np.mean(signal)
        Y = np.cumsum(signal - mean)
        R = np.max(Y) - np.min(Y)
        S = np.std(signal)
        if S > 0:
            features['nl_hurst_exponent'] = np.log(R / S) / np.log(N)
        else:
            features['nl_hurst_exponent'] = 0.5
    except:
        features['nl_hurst_exponent'] = 0.5
    
    # REMOVED: Lempel-Ziv Complexity - nested loops, too slow
    
    # Permutation Entropy
    try:
        m = 3  # embedding dimension
        tau = 1  # time delay
        n = len(signal)
        permutations = {}
        for i in range(n - (m - 1) * tau):
            # Extract embedded vector
            vec = [signal[i + j * tau] for j in range(m)]
            # Get permutation pattern
            perm = tuple(np.argsort(vec))
            permutations[perm] = permutations.get(perm, 0) + 1
        
        # Calculate entropy
        total = sum(permutations.values())
        probs = [count / total for count in permutations.values()]
        features['nl_permutation_entropy'] = -np.sum([p * np.log2(p) for p in probs if p > 0])
    except:
        features['nl_permutation_entropy'] = 0.0
    
    # Signal Mobility and Complexity (rate of change)
    try:
        d1 = np.diff(signal)
        d2 = np.diff(d1)
        var_signal = np.var(signal)
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)
        
        if var_signal > 0:
            mobility = np.sqrt(var_d1 / var_signal)
            features['nl_mobility'] = mobility
        else:
            features['nl_mobility'] = 0.0
        
        if var_d1 > 0 and mobility > 0:
            complexity = np.sqrt(var_d2 / var_d1) / mobility
            features['nl_complexity'] = complexity
        else:
            features['nl_complexity'] = 0.0
    except:
        features['nl_mobility'] = 0.0
        features['nl_complexity'] = 0.0
    
    return features

