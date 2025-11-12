# EOG Feature Extraction — Comprehensive Technical Documentation

**Project:** Fatigue Detection in Anesthesiologists Using EOG Signals  
**Author:** Yaniv Grosberg  
**Last Updated:** November 2025

---

## 📋 Overview

This document provides **complete technical details** for all **61 features** extracted from 16-second EOG windows.

### Feature Breakdown:
- **Time-Domain**: 26 features
- **Frequency-Domain**: 16 features  
- **Non-Linear**: 7 features
- **EOG-Specific**: 12 features
- **Total**: **61 features**

### Input Signal:
- **Window Size**: 16 seconds = 3,200 samples @ 200 Hz
- **Channels**: LOC (Left EOG), ROC (Right EOG)
- **Differential**: `diff_signal = LOC - ROC` (horizontal eye movements)

---

# 🧭 1. TIME-DOMAIN FEATURES (26)

Time-domain features analyze the EOG waveform directly without frequency transformation.

---

## 1.1 Basic Statistics (7 features)

### `td_mean` — Signal Mean
**Formula:**
```
mean = (1/N) × Σ(x[i])
```

**Computation:**
```python
td_mean = np.mean(signal)
```

**Interpretation:**
- **Normal**: Near 0 (after preprocessing removes DC offset)
- **Fatigue**: May drift due to electrode impedance changes
- **Clinical**: Not highly discriminative, but baseline reference

---

### `td_std` — Standard Deviation
**Formula:**
```
std = sqrt((1/N) × Σ(x[i] - mean)²)
```

**Computation:**
```python
td_std = np.std(signal)
```

**Interpretation:**
- **High**: Active eye movements (alert)
- **Low**: Reduced activity (fatigue)
- **Typical Alert**: 15-40 µV
- **Typical Drowsy**: 5-15 µV

---

### `td_var` — Variance
**Formula:**
```
var = std²
```

**Computation:**
```python
td_var = np.var(signal)
```

**Interpretation:**
- Quadratic measure of dispersion
- More sensitive to outliers than std
- **Fatigue**: Significant decrease (50-70%)

---

### `td_min`, `td_max` — Extrema
**Computation:**
```python
td_min = np.min(signal)
td_max = np.max(signal)
```

**Interpretation:**
- Capture peak deflections
- **Blinks**: typically produce extrema
- **Fatigue**: Reduced extrema magnitude

---

### `td_range` — Signal Range
**Formula:**
```
range = max - min
```

**Computation:**
```python
td_range = td_max - td_min
```

**Interpretation:**
- Total amplitude span
- **Alert**: 80-200 µV (with blinks)
- **Drowsy**: 20-80 µV (fewer/weaker blinks)

---

### `td_median` — Median Value
**Computation:**
```python
td_median = np.median(signal)
```

**Interpretation:**
- Robust central tendency (immune to outliers)
- Similar to mean for symmetric distributions
- Useful when blinks create extreme values

---

## 1.2 Shape Features (8 features)

### `td_peak_amp`, `td_valley_amp` — Peak/Valley Amplitudes
**Computation:**
```python
td_peak_amp = np.max(signal)
td_valley_amp = np.min(signal)
```

**Interpretation:**
- **Peak**: Maximum positive deflection
- **Valley**: Maximum negative deflection
- **Fatigue**: Both decrease as movements weaken

---

### `td_peak_pos_norm`, `td_valley_pos_norm` — Normalized Positions
**Formula:**
```
peak_pos_norm = argmax(signal) / N
valley_pos_norm = argmin(signal) / N
```

**Computation:**
```python
peak_idx = np.argmax(signal)
td_peak_pos_norm = peak_idx / len(signal)  # 0.0 to 1.0
```

**Interpretation:**
- **0.0**: Peak at window start
- **0.5**: Peak at window center
- **1.0**: Peak at window end
- **Clinical**: Temporal distribution of eye events

---

### `td_rise_time`, `td_fall_time` — Rise/Fall Times (Normalized)
**Computation:**
```python
# Rise time: samples until first peak occurrence
rise_time_samples = np.where(signal >= td_peak_amp)[0][0]
td_rise_time = rise_time_samples / len(signal)

# Fall time: samples from last peak occurrence to end
fall_time_samples = len(signal) - np.where(signal[::-1] >= td_peak_amp)[0][0]
td_fall_time = fall_time_samples / len(signal)
```

**Interpretation:**
- **Quick rise/fall**: Sharp blinks (alert)
- **Slow rise/fall**: Sluggish movements (fatigue)
- **Fatigue**: Increased rise time (slower initiation)

---

### `td_n_inflections` — Number of Inflection Points
**Formula:**
```
Inflection point: where signal curvature changes sign
curvature = d²x/dt² (second derivative)
```

**Computation:**
```python
slope = np.gradient(signal)
curvature = np.gradient(slope)
inflections = np.where(np.diff(np.sign(curvature)))[0]
td_n_inflections = len(inflections)
```

**Interpretation:**
- **High**: Complex waveform with many direction changes
- **Low**: Simple, monotonic movements (fatigue)
- **Alert**: 50-150 inflections/16s
- **Drowsy**: 10-50 inflections/16s

---

### `td_min_max_ratio` — Min/Max Ratio
**Formula:**
```
ratio = min / max
```

**Computation:**
```python
if td_peak_amp != 0:
    td_min_max_ratio = td_valley_amp / td_peak_amp
else:
    td_min_max_ratio = 0.0
```

**Interpretation:**
- **Negative**: Asymmetric (normal for EOG)
- **Near -1**: Symmetric positive/negative deflections
- **Near 0**: Dominant positive or negative deflections

---

## 1.3 Statistical Moments (2 features)

### `td_skewness` — Distribution Skewness
**Formula:**
```
skewness = E[(X - μ)³] / σ³
```

**Computation:**
```python
from scipy.stats import skew
td_skewness = skew(signal)
```

**Interpretation:**
- **> 0**: Right-skewed (long right tail)
- **< 0**: Left-skewed (long left tail)
- **≈ 0**: Symmetric distribution
- **Clinical**: Blinks create positive skew; fatigue may reduce asymmetry

---

### `td_kurtosis` — Distribution Kurtosis
**Formula:**
```
kurtosis = E[(X - μ)⁴] / σ⁴ - 3  (excess kurtosis)
```

**Computation:**
```python
from scipy.stats import kurtosis
td_kurtosis = kurtosis(signal)  # Fisher=True (excess)
```

**Interpretation:**
- **> 0**: Heavy tails (leptokurtic) - many extreme values
- **< 0**: Light tails (platykurtic) - few extreme values
- **≈ 0**: Normal distribution
- **Alert**: Positive (blinks create outliers)
- **Drowsy**: Negative (flatter distribution)

---

## 1.4 Signal Characteristics (6 features)

### `td_zero_crossing_rate` — Zero-Crossing Rate
**Formula:**
```
ZCR = (1/N) × Σ |sign(x[i]) - sign(x[i-1])|
```

**Computation:**
```python
zero_crossings = np.where(np.diff(np.sign(signal)))[0]
td_zero_crossing_rate = len(zero_crossings) / len(signal)
```

**Interpretation:**
- Measures oscillation frequency
- **High**: Rapid oscillations (alert, active scanning)
- **Low**: Slow drifts (drowsy, fixation)
- **Alert**: 0.02-0.08 (64-256 crossings/16s)
- **Drowsy**: 0.005-0.02 (16-64 crossings/16s)

---

### `td_rms` — Root Mean Square
**Formula:**
```
RMS = sqrt((1/N) × Σ x[i]²)
```

**Computation:**
```python
td_rms = np.sqrt(np.mean(np.square(signal)))
```

**Interpretation:**
- Energy measure (always positive)
- Similar to std but not mean-centered
- **Alert**: 20-50 µV
- **Drowsy**: 5-20 µV

---

### `td_crest_factor` — Crest Factor
**Formula:**
```
Crest Factor = peak / RMS
```

**Computation:**
```python
if td_rms != 0:
    td_crest_factor = td_peak_amp / td_rms
else:
    td_crest_factor = 0.0
```

**Interpretation:**
- Ratio of peak to average energy
- **High**: Sharp, impulsive events (blinks)
- **Low**: Smooth, continuous signal
- **With Blinks**: 5-15
- **Without Blinks**: 2-5

---

### `td_mean_abs_dev` — Mean Absolute Deviation
**Formula:**
```
MAD = (1/N) × Σ |x[i] - mean|
```

**Computation:**
```python
td_mean_abs_dev = np.mean(np.abs(signal - td_mean))
```

**Interpretation:**
- Robust dispersion measure
- Less sensitive to outliers than std
- **Fatigue**: Decreases (reduced variability)

---

### `td_energy` — Total Signal Energy
**Formula:**
```
Energy = Σ x[i]²
```

**Computation:**
```python
td_energy = np.sum(np.square(signal))
```

**Interpretation:**
- Cumulative power measure
- **Alert**: High (active movements)
- **Drowsy**: Low (minimal activity)
- **Typical Alert**: 1e6 - 1e8
- **Typical Drowsy**: 1e4 - 1e6

---

### `td_auc` — Area Under Curve
**Formula:**
```
AUC = ∫|x(t)| dt ≈ Σ |x[i]| × Δt
```

**Computation:**
```python
td_auc = np.trapz(np.abs(signal))
```

**Interpretation:**
- Total signal magnitude (time-weighted)
- Uses trapezoidal integration
- **Fatigue**: Reduced AUC (less overall movement)

---

## 1.5 Hjorth Parameters (3 features)

Hjorth parameters characterize signal dynamics in time domain.

### `td_hjorth_activity` — Activity
**Formula:**
```
Activity = var(x) = σ²
```

**Computation:**
```python
td_hjorth_activity = np.var(signal)
```

**Interpretation:**
- Signal power (variance)
- **Alert**: High activity
- **Drowsy**: Low activity

---

### `td_hjorth_mobility` — Mobility
**Formula:**
```
Mobility = σ(dx/dt) / σ(x)
```

**Computation:**
```python
signal_diff = np.diff(signal)
if td_std != 0:
    td_hjorth_mobility = np.std(signal_diff) / td_std
else:
    td_hjorth_mobility = 0.0
```

**Interpretation:**
- Mean frequency (rate of change)
- **High**: Fast oscillations
- **Low**: Slow drifts
- **Alert**: 0.5-2.0
- **Drowsy**: 0.1-0.5

---

### `td_hjorth_complexity` — Complexity
**Formula:**
```
Complexity = Mobility(dx/dt) / Mobility(x)
```

**Computation:**
```python
signal_diff2 = np.diff(signal_diff)
if td_hjorth_mobility != 0 and np.std(signal_diff) != 0:
    mobility_of_deriv = np.std(signal_diff2) / np.std(signal_diff)
    td_hjorth_complexity = mobility_of_deriv / td_hjorth_mobility
else:
    td_hjorth_complexity = 0.0
```

**Interpretation:**
- Deviation from sine wave
- **High**: Complex, irregular waveform
- **Low**: Simple, sine-like waveform
- **Alert**: 1.5-3.0 (complex patterns)
- **Drowsy**: 0.5-1.5 (simple oscillations)

---

# ⚡ 2. FREQUENCY-DOMAIN FEATURES (16)

Frequency-domain features use **Welch's Power Spectral Density** to analyze frequency content.

## Method: Welch's PSD

**Purpose:** Estimate power distribution across frequencies

**Steps:**
1. Divide signal into overlapping segments
2. Apply window function (Hanning) to each segment
3. Compute FFT for each segment
4. Average power spectra
5. Extract features from averaged PSD

**Code:**
```python
from scipy.signal import welch

fs = 200  # sampling frequency
nperseg = min(len(signal), fs * 2)  # 2-second segments
f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
# f: frequency bins (Hz)
# Pxx: power spectral density (µV²/Hz)
```

---

## 2.1 Absolute Band Powers (4 features)

### Band Definitions:
- **Delta (δ)**: 0.5-4 Hz — Very slow oscillations
- **Theta (θ)**: 4-8 Hz — **Drowsiness marker** ⭐
- **Alpha (α)**: 8-13 Hz — Relaxed wakefulness
- **Beta (β)**: 13-30 Hz — Active cognition

### Computation (Band Power):
**Formula:**
```
BP_band = ∫[f_low to f_high] PSD(f) df
```

**Code:**
```python
def bandpower(signal, fs, band):
    f, Pxx = welch(signal, fs=fs, nperseg=min(len(signal), fs*2))
    freq_res = f[1] - f[0]
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], dx=freq_res)

fd_bp_delta = bandpower(signal, 200, (0.5, 4))
fd_bp_theta = bandpower(signal, 200, (4, 8))
fd_bp_alpha = bandpower(signal, 200, (8, 13))
fd_bp_beta = bandpower(signal, 200, (13, 30))
```

**Interpretation:**
- **Delta**: Artifacts, very slow drifts
- **Theta**: **Increases with drowsiness** (↑↑)
- **Alpha**: Decreases with drowsiness (↓)
- **Beta**: Decreases with drowsiness (↓↓)

---

## 2.2 Relative Band Powers (4 features)

### `fd_rel_*` — Normalized Band Powers
**Formula:**
```
rel_band = BP_band / BP_total
where BP_total = Σ BP_all_bands
```

**Code:**
```python
total_power = bandpower(signal, 200, (0.5, 30))
fd_rel_delta = fd_bp_delta / total_power
fd_rel_theta = fd_bp_theta / total_power
fd_rel_alpha = fd_bp_alpha / total_power
fd_rel_beta = fd_bp_beta / total_power
```

**Interpretation:**
- Removes absolute power variability
- **Alert**: rel_beta > rel_theta
- **Drowsy**: rel_theta > rel_beta ⭐

---

## 2.3 Band Power Ratios (4 features)

### `fd_theta_alpha_ratio` — **Gold Standard for Drowsiness** ⭐
**Formula:**
```
θ/α = BP_theta / BP_alpha
```

**Code:**
```python
if fd_bp_alpha > 0:
    fd_theta_alpha_ratio = fd_bp_theta / fd_bp_alpha
else:
    fd_theta_alpha_ratio = 0.0
```

**Interpretation:**
- **Most important drowsiness marker!**
- **Alert**: 0.5-1.5
- **Transitional**: 1.5-3.0
- **Drowsy**: 3.0-10.0+ ⭐
- **Mechanism**: Theta increases as cortical arousal decreases

---

### `fd_delta_beta_ratio` — Low/High Frequency Ratio
**Formula:**
```
δ/β = BP_delta / BP_beta
```

**Interpretation:**
- **Alert**: < 1.0 (high-frequency dominance)
- **Drowsy**: > 2.0 (low-frequency dominance)

---

### `fd_low_high_ratio` — Combined Low/High Ratio
**Formula:**
```
(δ+θ)/(α+β) = (BP_delta + BP_theta) / (BP_alpha + BP_beta)
```

**Interpretation:**
- Broad-spectrum drowsiness measure
- **Alert**: < 0.5
- **Drowsy**: > 1.5

---

### `fd_alpha_beta_ratio` — Relaxation vs. Activation
**Formula:**
```
α/β = BP_alpha / BP_beta
```

**Interpretation:**
- **High**: Relaxed but awake
- **Low**: Active cognition or drowsiness

---

## 2.4 Spectral Characteristics (4 features)

### `fd_spectral_centroid` — Spectral Center of Mass
**Formula:**
```
Centroid = Σ(f × PSD(f)) / Σ(PSD(f))
```

**Code:**
```python
f, Pxx = welch(signal, fs=200, nperseg=min(len(signal), 400))
if np.sum(Pxx) > 0:
    fd_spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)
else:
    fd_spectral_centroid = 0.0
```

**Interpretation:**
- "Average frequency" weighted by power
- **Alert**: 15-25 Hz (high frequencies)
- **Drowsy**: 5-10 Hz (low frequencies) ⭐

---

### `fd_spectral_spread` — Spectral Standard Deviation
**Formula:**
```
Spread = sqrt(Σ((f - Centroid)² × PSD(f)) / Σ(PSD(f)))
```

**Code:**
```python
if np.sum(Pxx) > 0:
    fd_spectral_spread = np.sqrt(
        np.sum(((f - fd_spectral_centroid)**2) * Pxx) / np.sum(Pxx)
    )
else:
    fd_spectral_spread = 0.0
```

**Interpretation:**
- Bandwidth around centroid
- **High**: Broad-spectrum activity
- **Low**: Narrow-band activity (drowsiness)

---

### `fd_spectral_entropy` — Shannon Entropy of Spectrum
**Formula:**
```
H = -Σ p(f) × log₂(p(f))
where p(f) = PSD(f) / Σ(PSD(f))
```

**Code:**
```python
from scipy.stats import entropy

psd_norm = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else Pxx
psd_norm = psd_norm[psd_norm > 0]  # remove zeros
fd_spectral_entropy = entropy(psd_norm, base=2)
```

**Interpretation:**
- Measures spectral complexity/randomness
- **High**: Broad, uniform spectrum (alert)
- **Low**: Concentrated spectrum (drowsy)
- **Alert**: 6-8 bits
- **Drowsy**: 3-5 bits

---

### `fd_spectral_edge` — 95% Spectral Edge Frequency
**Formula:**
```
SEF95 = frequency where cumulative power = 95% of total
```

**Code:**
```python
cumsum = np.cumsum(Pxx)
if cumsum[-1] > 0:
    idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
    if len(idx) > 0:
        fd_spectral_edge = f[idx[0]]
    else:
        fd_spectral_edge = f[-1]
else:
    fd_spectral_edge = 0.0
```

**Interpretation:**
- Upper frequency bound of activity
- **Alert**: 25-40 Hz
- **Drowsy**: 10-20 Hz ⭐

---

# 🧩 3. NON-LINEAR FEATURES (7)

Non-linear features capture complexity, chaos, and fractal properties.

**Note:** We removed Sample Entropy, Approximate Entropy, DFA, and Lempel-Ziv Complexity because they are O(N²) algorithms that were **too slow** (30-40 seconds per window with 3,200 samples).

---

## 3.1 Fractal Dimension Measures (3 features)

### `nl_higuchi_fd` — Higuchi Fractal Dimension
**Purpose:** Measure self-similarity and waveform complexity

**Method:**
1. Construct k-max time series from original signal
2. Calculate length of each reconstructed curve
3. Compute fractal dimension from log-log slope

**Formula:**
```
For each k ∈ [1, k_max]:
  L(k) = average curve length for scale k
FD = slope of log(L(k)) vs log(1/k)
```

**Code:**
```python
def higuchi_fractal_dimension(signal, kmax=10):
    N = len(signal)
    L = []
    x = []
    
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            Lmk = 0
            for i in range(1, int((N - m) / k)):
                Lmk += abs(signal[m + i*k] - signal[m + (i-1)*k])
            Lmk = Lmk * (N - 1) / (k * k * int((N - m) / k))
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
        x.append(np.log(1.0 / k))
    
    # Linear regression
    hfd = np.polyfit(x, L, 1)[0]
    return hfd

nl_higuchi_fd = higuchi_fractal_dimension(signal, kmax=10)
```

**Interpretation:**
- **1.0-1.3**: Very smooth, regular signal (sine wave-like)
- **1.3-1.7**: Normal physiological complexity ⭐
- **1.7-2.0**: Highly complex, irregular signal
- **Drowsiness**: ↓ Complexity (closer to 1.0)

---

### `nl_petrosian_fd` — Petrosian Fractal Dimension
**Purpose:** Fast approximation based on sign changes

**Formula:**
```
PFD = log₁₀(N) / [log₁₀(N) + log₁₀(N / (N + 0.4 × N_Δ))]

where N_Δ = number of sign changes in first derivative
```

**Code:**
```python
N = len(signal)
diff = np.diff(signal)
N_delta = len(np.where(diff[:-1] * diff[1:] < 0)[0])  # sign changes

if N_delta > 0:
    nl_petrosian_fd = np.log10(N) / (
        np.log10(N) + np.log10(N / (N + 0.4 * N_delta))
    )
else:
    nl_petrosian_fd = 1.0
```

**Interpretation:**
- Similar to Higuchi FD but faster
- **Alert**: 1.03-1.08 (many sign changes)
- **Drowsy**: 1.00-1.03 (few sign changes)

---

### `nl_katz_fd` — Katz Fractal Dimension
**Purpose:** Geometric measure based on curve length vs. diameter

**Formula:**
```
KFD = log₁₀(L) / log₁₀(d)

where:
  L = total curve length = Σ |diff(signal)|
  d = max distance from first point = max|signal - signal[0]|
```

**Code:**
```python
dists = np.abs(np.diff(signal))
L = np.sum(dists)
d = np.max(np.abs(signal - signal[0]))

if d > 0 and L > 0:
    nl_katz_fd = np.log10(L) / np.log10(d)
else:
    nl_katz_fd = 1.0
```

**Interpretation:**
- **High L, small d**: Complex path (high FD)
- **Low L, large d**: Simple path (low FD)
- **Alert**: 5-15
- **Drowsy**: 1-5

---

## 3.2 Long-Range Correlation (1 feature)

### `nl_hurst_exponent` — Hurst Exponent (Simplified R/S)
**Purpose:** Measure persistence vs. randomness

**Formula:**
```
H = log(R/S) / log(N)

where:
  R = range of cumulative deviations = max(Y) - min(Y)
  S = standard deviation of signal
  Y = cumulative sum of mean-centered signal
```

**Code:**
```python
N = len(signal)
mean = np.mean(signal)
Y = np.cumsum(signal - mean)  # integrate
R = np.max(Y) - np.min(Y)     # range
S = np.std(signal)             # std

if S > 0:
    nl_hurst_exponent = np.log(R / S) / np.log(N)
else:
    nl_hurst_exponent = 0.5
```

**Interpretation:**
- **H = 0.5**: Random walk (white noise)
- **H > 0.5**: Persistent (trend-reinforcing) ⭐
- **H < 0.5**: Anti-persistent (mean-reverting)
- **Alert**: 0.6-0.8 (complex patterns)
- **Drowsy**: 0.3-0.5 (more random)

---

## 3.3 Entropy Measures (1 feature)

### `nl_permutation_entropy` — Permutation Entropy
**Purpose:** Quantify ordinal pattern complexity

**Method:**
1. Embed signal in m-dimensional space
2. For each embedded vector, find permutation pattern
3. Count frequency of each permutation
4. Calculate Shannon entropy

**Formula:**
```
PE = -Σ p(π) × log₂(p(π))

where π = permutation patterns
```

**Code:**
```python
m = 3  # embedding dimension
tau = 1  # time delay
n = len(signal)
permutations = {}

for i in range(n - (m - 1) * tau):
    vec = [signal[i + j * tau] for j in range(m)]
    perm = tuple(np.argsort(vec))  # ordinal pattern
    permutations[perm] = permutations.get(perm, 0) + 1

# Calculate entropy
total = sum(permutations.values())
probs = [count / total for count in permutations.values()]
nl_permutation_entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
```

**Interpretation:**
- **High**: Many diverse patterns (complex, alert)
- **Low**: Few repetitive patterns (simple, drowsy)
- **Max entropy (m=3)**: log₂(3!) = 2.58 bits
- **Alert**: 2.2-2.5 bits
- **Drowsy**: 1.5-2.0 bits ⭐

---

## 3.4 Derivative-Based Complexity (2 features)

### `nl_mobility` — Signal Mobility
**Formula:**
```
Mobility = sqrt(var(dx/dt) / var(x))
```

**Code:**
```python
d1 = np.diff(signal)
var_signal = np.var(signal)
var_d1 = np.var(d1)

if var_signal > 0:
    nl_mobility = np.sqrt(var_d1 / var_signal)
else:
    nl_mobility = 0.0
```

**Interpretation:**
- Rate of change (similar to Hjorth mobility)
- **High**: Fast fluctuations
- **Low**: Slow drifts
- **Drowsy**: ↓ Mobility

---

### `nl_complexity` — Signal Complexity
**Formula:**
```
Complexity = Mobility(dx/dt) / Mobility(x)
```

**Code:**
```python
d2 = np.diff(d1)
var_d2 = np.var(d2)

if var_d1 > 0 and nl_mobility > 0:
    mobility_of_deriv = np.sqrt(var_d2 / var_d1)
    nl_complexity = mobility_of_deriv / nl_mobility
else:
    nl_complexity = 0.0
```

**Interpretation:**
- Deviation from sine wave
- **High**: Complex waveform
- **Low**: Simple oscillation
- **Drowsy**: ↓ Complexity ⭐

---

# 👁️ 4. EOG-SPECIFIC FEATURES (12)

Features designed specifically for eye movement analysis.

---

## 4.1 Blink Detection & Characteristics (3 features)

### Blink Detection Algorithm

**Method:** Find peaks in EOG signal using SciPy

**Code:**
```python
from scipy.signal import find_peaks

def detect_blinks(signal, fs=200):
    min_amplitude = 1.5 * np.std(signal)
    min_distance = int(0.2 * fs)  # 200ms between blinks
    
    # Find positive and negative peaks
    pos_peaks, pos_props = find_peaks(
        signal, 
        height=min_amplitude, 
        distance=min_distance
    )
    neg_peaks, neg_props = find_peaks(
        -signal, 
        height=min_amplitude, 
        distance=min_distance
    )
    
    # Combine
    all_peaks = np.concatenate([pos_peaks, neg_peaks])
    all_amps = np.concatenate([
        pos_props['peak_heights'], 
        neg_props['peak_heights']
    ])
    
    return all_peaks, all_amps
```

---

### `eog_blink_rate` — Blink Frequency
**Formula:**
```
Blink Rate = N_blinks / T
where T = window duration in seconds
```

**Code:**
```python
blinks, amps = detect_blinks(loc_signal, fs=200)
duration = len(loc_signal) / 200  # seconds
eog_blink_rate = len(blinks) / duration  # blinks/second
```

**Interpretation:**
- **Normal Alert**: 0.2-0.4 blinks/sec (3-6 blinks/16s)
- **Drowsy**: 0.05-0.15 blinks/sec (1-2 blinks/16s) ⭐
- **Mechanism**: Reduced cortical arousal → fewer spontaneous blinks

---

### `eog_blink_amp_mean` — Mean Blink Amplitude
**Code:**
```python
if len(amps) > 0:
    eog_blink_amp_mean = np.mean(amps)
else:
    eog_blink_amp_mean = 0.0
```

**Interpretation:**
- **Alert**: 50-150 µV
- **Drowsy**: 20-80 µV ⭐
- **Fatigue**: Weaker eye muscle contractions

---

### `eog_blink_amp_std` — Blink Amplitude Variability
**Code:**
```python
if len(amps) > 0:
    eog_blink_amp_std = np.std(amps)
else:
    eog_blink_amp_std = 0.0
```

**Interpretation:**
- **Consistent blinks**: Low std (alert)
- **Irregular blinks**: High std (fatigue/distraction)

---

## 4.2 LOC-ROC Synchronization (3 features)

### `eog_loc_roc_corr` — Pearson Correlation
**Formula:**
```
r = Σ((LOC - LOC_mean) × (ROC - ROC_mean)) / (σ_LOC × σ_ROC × N)
```

**Code:**
```python
from scipy.stats import pearsonr

try:
    corr, p_value = pearsonr(loc_signal, roc_signal)
    eog_loc_roc_corr = corr
except:
    eog_loc_roc_corr = 0.0
```

**Interpretation:**
- **High (0.7-0.9)**: Synchronized binocular movements (alert)
- **Low (0.3-0.6)**: Desynchronized (fatigue/asymmetry) ⭐
- **Negative**: Anti-correlated (rare, artifact)

---

### `eog_loc_roc_lag` — Cross-Correlation Lag
**Purpose:** Measure phase difference between eyes

**Formula:**
```
Lag = argmax(Cross-Correlation(LOC, ROC))
```

**Code:**
```python
from scipy.signal import correlate

# Center signals
loc_centered = loc_signal - np.mean(loc_signal)
roc_centered = roc_signal - np.mean(roc_signal)

# Cross-correlation
cross_corr = correlate(loc_centered, roc_centered, mode='same')
lag_samples = np.argmax(np.abs(cross_corr)) - len(cross_corr) // 2
eog_loc_roc_lag = lag_samples / fs  # convert to seconds
```

**Interpretation:**
- **Near 0**: Synchronous movements
- **> 50ms**: Significant delay (possible fatigue/pathology)

---

### `eog_loc_roc_lag_corr` — Lag Correlation Magnitude
**Code:**
```python
eog_loc_roc_lag_corr = np.max(np.abs(cross_corr)) / len(loc_signal)
```

**Interpretation:**
- Strength of cross-correlation at optimal lag
- **High**: Strong bilateral coordination
- **Low**: Weak coordination

---

## 4.3 Eye Movement Dynamics (4 features)

### `eog_movement_amplitude` — Horizontal Movement Range
**Formula:**
```
Amplitude = range(LOC - ROC)
```

**Code:**
```python
diff_signal = loc_signal - roc_signal
eog_movement_amplitude = np.max(diff_signal) - np.min(diff_signal)
```

**Interpretation:**
- Measures lateral eye excursions
- **Alert**: 100-300 µV (active scanning)
- **Drowsy**: 20-100 µV (minimal movement) ⭐

---

### `eog_movement_velocity_mean` — Mean Eye Movement Velocity ⭐
**Formula:**
```
Velocity = |d(LOC - ROC)/dt|
```

**Code:**
```python
velocity = np.abs(np.diff(diff_signal))
eog_movement_velocity_mean = np.mean(velocity)
```

**Interpretation:**
- **Critical fatigue marker!**
- **Alert**: 0.5-2.0 µV/sample (100-400 µV/sec)
- **Drowsy**: 0.1-0.5 µV/sample (20-100 µV/sec) ⭐
- **Mechanism**: Slower saccades, more fixations

---

### `eog_movement_velocity_std` — Velocity Variability
**Code:**
```python
eog_movement_velocity_std = np.std(velocity)
```

**Interpretation:**
- **High**: Mix of fast/slow movements (alert)
- **Low**: Uniform slow movements (drowsy)

---

### `eog_saccade_rate` — Saccade Frequency
**Purpose:** Detect rapid eye movements (saccades)

**Method:**
```
Saccade = high-velocity peak in differential EOG
Threshold = mean(velocity) + 2 × std(velocity)
```

**Code:**
```python
velocity_threshold = np.mean(velocity) + 2 * np.std(velocity)
saccades, _ = find_peaks(
    velocity, 
    height=velocity_threshold, 
    distance=int(0.02 * fs)  # 20ms minimum between saccades
)
eog_saccade_rate = len(saccades) / (len(loc_signal) / fs)
```

**Interpretation:**
- **Alert**: 2-5 saccades/sec (active visual scanning)
- **Drowsy**: 0.2-1 saccades/sec (fixation, staring) ⭐
- **Mechanism**: Reduced attentional shifts

---

## 4.4 Bilateral Asymmetry (2 features)

### `eog_loc_roc_energy_ratio` — Energy Ratio
**Formula:**
```
Energy Ratio = Σ(LOC²) / Σ(ROC²)
```

**Code:**
```python
eog_loc_roc_energy_ratio = (
    np.sum(loc_signal ** 2) / (np.sum(roc_signal ** 2) + 1e-10)
)
```

**Interpretation:**
- **≈ 1.0**: Symmetric activity (normal)
- **> 1.5 or < 0.67**: Asymmetric (possible fatigue or pathology)

---

### `eog_loc_roc_amplitude_ratio` — Amplitude Ratio
**Formula:**
```
Amplitude Ratio = max|LOC| / max|ROC|
```

**Code:**
```python
eog_loc_roc_amplitude_ratio = (
    np.max(np.abs(loc_signal)) / (np.max(np.abs(roc_signal)) + 1e-10)
)
```

**Interpretation:**
- **≈ 1.0**: Symmetric peak amplitudes
- **Deviation**: Asymmetric movements (monitor for pathology)

---

# 📊 SUMMARY & CLINICAL INTERPRETATION

## Feature Importance Ranking (Based on Physiology)

### 🔴 **Critical Features (Top Priority)**

1. **`fd_theta_alpha_ratio`** — Gold standard drowsiness marker
2. **`eog_movement_velocity_mean`** — Direct fatigue indicator
3. **`eog_saccade_rate`** — Attentional engagement
4. **`fd_rel_theta`** — Theta power increase
5. **`fd_spectral_centroid`** — Frequency shift to lower bands

### 🟡 **Secondary Features**

6. **`eog_blink_rate`** — Arousal level
7. **`fd_spectral_edge`** — High-frequency activity
8. **`td_std`** — Overall activity level
9. **`nl_higuchi_fd`** — Signal complexity
10. **`eog_loc_roc_corr`** — Binocular coordination

### 🟢 **Supporting Features**

- All other time-domain statistics
- Non-linear complexity measures
- Asymmetry indicators

---

## Expected Changes with Fatigue

| Feature Category | Direction | Magnitude | Mechanism |
|-----------------|-----------|-----------|-----------|
| **Theta Power** | ↑↑↑ | 2-5× | Reduced cortical arousal |
| **Alpha/Beta Power** | ↓↓ | 30-60% | Decreased vigilance |
| **θ/α Ratio** | ↑↑↑ | 2-10× | Classic drowsiness signature |
| **Movement Velocity** | ↓↓ | 50-70% | Slower saccades, more fixations |
| **Saccade Rate** | ↓↓ | 60-80% | Reduced attentional shifts |
| **Blink Rate** | ↓ | 30-50% | Lower arousal |
| **Signal Complexity** | ↓ | Variable | Simpler, repetitive patterns |
| **Spectral Centroid** | ↓↓ | 10-20 Hz → 5-10 Hz | Shift to low frequencies |

---

## Feature Extraction Summary

### Total Features: **61**

| Category | Count | Computation Time* |
|----------|-------|-------------------|
| Time-Domain | 26 | ~1-2 ms/window |
| Frequency-Domain | 16 | ~5-8 ms/window |
| Non-Linear | 7 | ~10-15 ms/window |
| EOG-Specific | 12 | ~3-5 ms/window |
| **Total** | **61** | **~20-30 ms/window** |

*Per 16-second window (3,200 samples @ 200 Hz)

---

## Implementation Notes

### Preprocessing (Applied Before Feature Extraction):
1. **Bandpass Filter**: 0.5-30 Hz (remove DC drift and high-frequency noise)
2. **Resampling**: Standardize to 200 Hz
3. **Artifact Rejection**: (Optional) Remove extreme outliers

### Windowing:
- **Size**: 16 seconds (3,200 samples)
- **Stride**: 8 seconds (50% overlap)
- **Justification**: Matches CNN_16s configuration

### Labeling:
- **Threshold**: ≥10% drowsy samples in window → label=1
- **Rationale**: Typical microsleep events are 0.5-3 seconds

---

## References

### Feature Computation:
- **Hjorth Parameters**: Hjorth, B. (1970). EEG analysis based on time domain properties.
- **Higuchi FD**: Higuchi, T. (1988). Approach to an irregular time series.
- **Permutation Entropy**: Bandt, C. & Pompe, B. (2002). Permutation entropy.
- **Welch's PSD**: Welch, P. (1967). The use of fast Fourier transform.

### Clinical Drowsiness Markers:
- **θ/α Ratio**: Lal & Craig (2001). Driver fatigue: EEG changes.
- **EOG Velocity**: Schleicher et al. (2008). Blinks and saccades as indicators.
- **Spectral Changes**: Jap et al. (2009). Using EEG spectral components.

---

**End of Documentation**

For implementation details, see:
- `time_domain.py`
- `frequency_domain.py`
- `nonlinear.py`
- `eog_specific.py`
