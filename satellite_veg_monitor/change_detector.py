"""
Module 3: Temporal Change Detection
Analyze vegetation changes between two time periods using spectral indices.

Author: Mohmmad Umayr Romshoo
"""

import numpy as np
from typing import Dict, Tuple, Optional


def calculate_change(index_before: np.ndarray, index_after: np.ndarray) -> np.ndarray:
    """
    Calculate the difference between two spectral indices.
    
    Parameters:
    -----------
    index_before : np.ndarray
        Spectral index array from the earlier time period
    index_after : np.ndarray
        Spectral index array from the later time period
        
    Returns:
    --------
    np.ndarray
        Change array (after - before)
        Positive values indicate increase, negative indicate decrease
    """
    if index_before.shape != index_after.shape:
        raise ValueError("Input arrays must have the same shape")
    
    return index_after - index_before


def calculate_percent_change(index_before: np.ndarray, index_after: np.ndarray, 
                            epsilon: float = 1e-10) -> np.ndarray:
    """
    Calculate the percentage change between two spectral indices.
    
    Parameters:
    -----------
    index_before : np.ndarray
        Spectral index array from the earlier time period
    index_after : np.ndarray
        Spectral index array from the later time period
    epsilon : float
        Small value to prevent division by zero
        
    Returns:
    --------
    np.ndarray
        Percentage change array
    """
    if index_before.shape != index_after.shape:
        raise ValueError("Input arrays must have the same shape")
    
    # Avoid division by zero
    denominator = np.where(np.abs(index_before) < epsilon, epsilon, index_before)
    percent_change = ((index_after - index_before) / denominator) * 100
    
    return percent_change


def calculate_change_magnitude(index_before: np.ndarray, index_after: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute magnitude of change.
    
    Parameters:
    -----------
    index_before : np.ndarray
        Spectral index array from the earlier time period
    index_after : np.ndarray
        Spectral index array from the later time period
        
    Returns:
    --------
    np.ndarray
        Absolute change magnitude
    """
    change = calculate_change(index_before, index_after)
    return np.abs(change)


def detect_significant_change(index_before: np.ndarray, index_after: np.ndarray,
                              threshold: float = 0.1) -> np.ndarray:
    """
    Detect pixels with significant change based on threshold.
    
    Parameters:
    -----------
    index_before : np.ndarray
        Spectral index array from the earlier time period
    index_after : np.ndarray
        Spectral index array from the later time period
    threshold : float
        Minimum change magnitude to be considered significant
        
    Returns:
    --------
    np.ndarray
        Boolean array where True indicates significant change
    """
    change_magnitude = calculate_change_magnitude(index_before, index_after)
    return change_magnitude >= threshold


def calculate_change_statistics(index_before: np.ndarray, index_after: np.ndarray,
                                mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive statistics about the change.
    
    Parameters:
    -----------
    index_before : np.ndarray
        Spectral index array from the earlier time period
    index_after : np.ndarray
        Spectral index array from the later time period
    mask : np.ndarray, optional
        Boolean mask to apply to the data (True = valid, False = ignore)
        
    Returns:
    --------
    dict
        Dictionary containing change statistics:
        - mean_change: Average change value
        - std_change: Standard deviation of change
        - min_change: Minimum change value
        - max_change: Maximum change value
        - mean_abs_change: Mean absolute change
        - percent_increased: Percentage of pixels that increased
        - percent_decreased: Percentage of pixels that decreased
        - percent_unchanged: Percentage of pixels with minimal change
    """
    change = calculate_change(index_before, index_after)
    
    # Apply mask if provided
    if mask is not None:
        change_masked = change[mask]
    else:
        change_masked = change.flatten()
    
    # Remove NaN values
    change_valid = change_masked[~np.isnan(change_masked)]
    
    if len(change_valid) == 0:
        raise ValueError("No valid pixels to analyze")
    
    stats = {
        'mean_change': float(np.mean(change_valid)),
        'std_change': float(np.std(change_valid)),
        'min_change': float(np.min(change_valid)),
        'max_change': float(np.max(change_valid)),
        'mean_abs_change': float(np.mean(np.abs(change_valid))),
        'percent_increased': float(np.sum(change_valid > 0.01) / len(change_valid) * 100),
        'percent_decreased': float(np.sum(change_valid < -0.01) / len(change_valid) * 100),
        'percent_unchanged': float(np.sum(np.abs(change_valid) <= 0.01) / len(change_valid) * 100)
    }
    
    return stats


def analyze_ndvi_change(ndvi_before: np.ndarray, ndvi_after: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Comprehensive NDVI change analysis.
    
    Parameters:
    -----------
    ndvi_before : np.ndarray
        NDVI array from the earlier time period
    ndvi_after : np.ndarray
        NDVI array from the later time period
    mask : np.ndarray, optional
        Boolean mask to apply to the data
        
    Returns:
    --------
    dict
        Dictionary containing:
        - change: Absolute change
        - percent_change: Percentage change
        - magnitude: Change magnitude
        - statistics: Change statistics
    """
    change = calculate_change(ndvi_before, ndvi_after)
    percent_change = calculate_percent_change(ndvi_before, ndvi_after)
    magnitude = calculate_change_magnitude(ndvi_before, ndvi_after)
    stats = calculate_change_statistics(ndvi_before, ndvi_after, mask)
    
    return {
        'change': change,
        'percent_change': percent_change,
        'magnitude': magnitude,
        'statistics': stats
    }


def analyze_all_indices_change(indices_before: Dict[str, np.ndarray],
                               indices_after: Dict[str, np.ndarray],
                               mask: Optional[np.ndarray] = None) -> Dict[str, Dict]:
    """
    Analyze changes for all spectral indices.
    
    Parameters:
    -----------
    indices_before : dict
        Dictionary of spectral indices from the earlier time period
        Keys: 'ndvi', 'evi', 'savi', 'ndwi', 'nbr'
    indices_after : dict
        Dictionary of spectral indices from the later time period
    mask : np.ndarray, optional
        Boolean mask to apply to the data
        
    Returns:
    --------
    dict
        Dictionary with change analysis for each index
    """
    results = {}
    
    for index_name in indices_before.keys():
        if index_name not in indices_after:
            continue
            
        before = indices_before[index_name]
        after = indices_after[index_name]
        
        results[index_name] = {
            'change': calculate_change(before, after),
            'percent_change': calculate_percent_change(before, after),
            'magnitude': calculate_change_magnitude(before, after),
            'statistics': calculate_change_statistics(before, after, mask)
        }
    
    return results


def identify_change_hotspots(index_before: np.ndarray, index_after: np.ndarray,
                             percentile: float = 90) -> np.ndarray:
    """
    Identify areas with extreme changes (hotspots).
    
    Parameters:
    -----------
    index_before : np.ndarray
        Spectral index array from the earlier time period
    index_after : np.ndarray
        Spectral index array from the later time period
    percentile : float
        Percentile threshold for identifying hotspots (default: 90)
        
    Returns:
    --------
    np.ndarray
        Boolean array where True indicates a hotspot
    """
    magnitude = calculate_change_magnitude(index_before, index_after)
    threshold = np.nanpercentile(magnitude, percentile)
    
    return magnitude >= threshold


def calculate_trend_direction(index_before: np.ndarray, index_after: np.ndarray,
                              no_change_threshold: float = 0.01) -> np.ndarray:
    """
    Classify the direction of change.
    
    Parameters:
    -----------
    index_before : np.ndarray
        Spectral index array from the earlier time period
    index_after : np.ndarray
        Spectral index array from the later time period
    no_change_threshold : float
        Threshold below which change is considered negligible
        
    Returns:
    --------
    np.ndarray
        Integer array with trend direction:
        1 = increase, 0 = no change, -1 = decrease
    """
    change = calculate_change(index_before, index_after)
    
    trend = np.zeros_like(change, dtype=np.int8)
    trend[change > no_change_threshold] = 1  # Increase
    trend[change < -no_change_threshold] = -1  # Decrease
    
    return trend
