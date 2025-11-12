"""
Feature Engineering Module for EOG-based Drowsiness Detection
==============================================================

This module provides comprehensive feature extraction for EOG signals,
organized into four categories:
    - Time-domain features
    - Frequency-domain features
    - Non-linear features
    - EOG-specific features

Author: Yaniv Grosberg
Date: 2025-11-09
"""

from .time_domain import extract_time_domain_features
from .frequency_domain import extract_frequency_domain_features
from .nonlinear import extract_nonlinear_features
from .eog_specific import extract_eog_specific_features

__all__ = [
    'extract_time_domain_features',
    'extract_frequency_domain_features',
    'extract_nonlinear_features',
    'extract_eog_specific_features'
]

