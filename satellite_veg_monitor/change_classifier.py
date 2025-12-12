"""
Module 4: Change Classification
Classify vegetation change severity with multiple thresholds.

Author: Mohmmad Umayr Romshoo
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import IntEnum


class ChangeClass(IntEnum):
    """Enumeration for change classification levels."""
    SEVERE_DECREASE = -3
    MODERATE_DECREASE = -2
    SLIGHT_DECREASE = -1
    NO_CHANGE = 0
    SLIGHT_INCREASE = 1
    MODERATE_INCREASE = 2
    SEVERE_INCREASE = 3


class VegetationHealthClass(IntEnum):
    """Enumeration for vegetation health classification."""
    DEGRADED = 1
    STRESSED = 2
    MODERATE = 3
    HEALTHY = 4
    VERY_HEALTHY = 5


def classify_change_magnitude(change: np.ndarray, 
                              thresholds: Optional[List[float]] = None) -> np.ndarray:
    """
    Classify change magnitude into discrete categories.
    
    Parameters:
    -----------
    change : np.ndarray
        Change array (difference between two time periods)
    thresholds : list of float, optional
        List of 3 thresholds for classification [slight, moderate, severe]
        Default: [0.05, 0.15, 0.30]
        
    Returns:
    --------
    np.ndarray
        Integer array with classification:
        -3: Severe decrease, -2: Moderate decrease, -1: Slight decrease
         0: No change
        +1: Slight increase, +2: Moderate increase, +3: Severe increase
    """
    if thresholds is None:
        thresholds = [0.05, 0.15, 0.30]
    
    if len(thresholds) != 3:
        raise ValueError("Exactly 3 thresholds required")
    
    t1, t2, t3 = sorted(thresholds)  # Ensure ascending order
    
    classification = np.zeros_like(change, dtype=np.int8)
    
    # Positive changes (increases)
    classification[(change >= t1) & (change < t2)] = ChangeClass.SLIGHT_INCREASE
    classification[(change >= t2) & (change < t3)] = ChangeClass.MODERATE_INCREASE
    classification[change >= t3] = ChangeClass.SEVERE_INCREASE
    
    # Negative changes (decreases)
    classification[(change <= -t1) & (change > -t2)] = ChangeClass.SLIGHT_DECREASE
    classification[(change <= -t2) & (change > -t3)] = ChangeClass.MODERATE_DECREASE
    classification[change <= -t3] = ChangeClass.SEVERE_DECREASE
    
    return classification


def classify_ndvi_health(ndvi: np.ndarray) -> np.ndarray:
    """
    Classify vegetation health based on NDVI values.
    
    Parameters:
    -----------
    ndvi : np.ndarray
        NDVI array
        
    Returns:
    --------
    np.ndarray
        Integer array with health classification:
        1: Degraded (NDVI < 0.2)
        2: Stressed (0.2 <= NDVI < 0.4)
        3: Moderate (0.4 <= NDVI < 0.6)
        4: Healthy (0.6 <= NDVI < 0.8)
        5: Very Healthy (NDVI >= 0.8)
    """
    health = np.zeros_like(ndvi, dtype=np.int8)
    
    health[ndvi < 0.2] = VegetationHealthClass.DEGRADED
    health[(ndvi >= 0.2) & (ndvi < 0.4)] = VegetationHealthClass.STRESSED
    health[(ndvi >= 0.4) & (ndvi < 0.6)] = VegetationHealthClass.MODERATE
    health[(ndvi >= 0.6) & (ndvi < 0.8)] = VegetationHealthClass.HEALTHY
    health[ndvi >= 0.8] = VegetationHealthClass.VERY_HEALTHY
    
    return health


def classify_change_with_context(change: np.ndarray,
                                 index_before: np.ndarray,
                                 index_after: np.ndarray,
                                 thresholds: Optional[List[float]] = None) -> np.ndarray:
    """
    Classify change considering the context of initial and final values.
    
    This provides more nuanced classification by considering:
    - Magnitude of change
    - Initial vegetation condition
    - Final vegetation condition
    
    Parameters:
    -----------
    change : np.ndarray
        Change array
    index_before : np.ndarray
        Index values before change
    index_after : np.ndarray
        Index values after change
    thresholds : list of float, optional
        Classification thresholds
        
    Returns:
    --------
    np.ndarray
        Integer array with contextual classification
    """
    # Base classification
    base_class = classify_change_magnitude(change, thresholds)
    
    # Adjust based on context
    contextual_class = base_class.copy()
    
    # If both before and after are very low (degraded), even small positive changes are significant
    low_both = (index_before < 0.3) & (index_after < 0.3)
    contextual_class[low_both & (change > 0.02)] = np.maximum(
        contextual_class[low_both & (change > 0.02)], 
        ChangeClass.SLIGHT_INCREASE
    )
    
    # If degradation from healthy to unhealthy, classify as more severe
    degradation = (index_before > 0.6) & (index_after < 0.4)
    contextual_class[degradation & (change < 0)] = np.minimum(
        contextual_class[degradation & (change < 0)],
        ChangeClass.MODERATE_DECREASE
    )
    
    return contextual_class


def classify_change_trajectory(indices_timeseries: List[np.ndarray]) -> np.ndarray:
    """
    Classify change trajectory over multiple time periods.
    
    Parameters:
    -----------
    indices_timeseries : list of np.ndarray
        List of index arrays in chronological order (minimum 3 required)
        
    Returns:
    --------
    np.ndarray
        Integer array with trajectory classification:
        -2: Consistent decline
        -1: Declining trend with fluctuations
         0: Stable/fluctuating
        +1: Increasing trend with fluctuations
        +2: Consistent growth
    """
    if len(indices_timeseries) < 3:
        raise ValueError("At least 3 time periods required for trajectory analysis")
    
    # Calculate changes between consecutive periods
    changes = [indices_timeseries[i+1] - indices_timeseries[i] 
               for i in range(len(indices_timeseries) - 1)]
    
    # Stack changes
    changes_stack = np.stack(changes, axis=-1)
    
    # Count positive and negative changes
    positive_count = np.sum(changes_stack > 0.05, axis=-1)
    negative_count = np.sum(changes_stack < -0.05, axis=-1)
    total_periods = len(changes)
    
    trajectory = np.zeros(changes_stack.shape[:-1], dtype=np.int8)
    
    # Consistent decline (most changes negative)
    trajectory[negative_count >= total_periods * 0.75] = -2
    
    # Declining trend (more negative than positive)
    trajectory[(negative_count > positive_count) & (trajectory == 0)] = -1
    
    # Consistent growth (most changes positive)
    trajectory[positive_count >= total_periods * 0.75] = 2
    
    # Increasing trend (more positive than negative)
    trajectory[(positive_count > negative_count) & (trajectory == 0)] = 1
    
    return trajectory


def classify_seasonal_change(change: np.ndarray,
                            expected_seasonal_change: float = 0.0) -> np.ndarray:
    """
    Classify change accounting for expected seasonal variations.
    
    Parameters:
    -----------
    change : np.ndarray
        Observed change array
    expected_seasonal_change : float
        Expected change due to seasonal effects
        
    Returns:
    --------
    np.ndarray
        Integer array with adjusted classification
    """
    # Subtract expected seasonal change
    adjusted_change = change - expected_seasonal_change
    
    # Classify adjusted change
    return classify_change_magnitude(adjusted_change)


def classify_all_changes(changes: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """
    Classify changes for all spectral indices.
    
    Parameters:
    -----------
    changes : dict
        Dictionary containing change analysis for each index
        (output from change_detector.analyze_all_indices_change)
        
    Returns:
    --------
    dict
        Dictionary with classifications for each index
    """
    classifications = {}
    
    # Define index-specific thresholds
    thresholds_map = {
        'ndvi': [0.05, 0.15, 0.30],
        'evi': [0.05, 0.15, 0.30],
        'savi': [0.05, 0.15, 0.30],
        'ndwi': [0.05, 0.15, 0.30],
        'nbr': [0.10, 0.25, 0.50]  # NBR typically has larger ranges
    }
    
    for index_name, change_data in changes.items():
        thresholds = thresholds_map.get(index_name, [0.05, 0.15, 0.30])
        
        classifications[index_name] = classify_change_magnitude(
            change_data['change'],
            thresholds
        )
    
    return classifications


def get_classification_statistics(classification: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate statistics for classification results.
    
    Parameters:
    -----------
    classification : np.ndarray
        Classification array
    mask : np.ndarray, optional
        Boolean mask to apply
        
    Returns:
    --------
    dict
        Dictionary with percentage of each classification class
    """
    if mask is not None:
        class_masked = classification[mask]
    else:
        class_masked = classification.flatten()
    
    total_pixels = len(class_masked)
    
    stats = {
        'severe_decrease': float(np.sum(class_masked == ChangeClass.SEVERE_DECREASE) / total_pixels * 100),
        'moderate_decrease': float(np.sum(class_masked == ChangeClass.MODERATE_DECREASE) / total_pixels * 100),
        'slight_decrease': float(np.sum(class_masked == ChangeClass.SLIGHT_DECREASE) / total_pixels * 100),
        'no_change': float(np.sum(class_masked == ChangeClass.NO_CHANGE) / total_pixels * 100),
        'slight_increase': float(np.sum(class_masked == ChangeClass.SLIGHT_INCREASE) / total_pixels * 100),
        'moderate_increase': float(np.sum(class_masked == ChangeClass.MODERATE_INCREASE) / total_pixels * 100),
        'severe_increase': float(np.sum(class_masked == ChangeClass.SEVERE_INCREASE) / total_pixels * 100),
        'total_decrease': float(np.sum(class_masked < 0) / total_pixels * 100),
        'total_increase': float(np.sum(class_masked > 0) / total_pixels * 100),
    }
    
    return stats


def create_change_severity_map(classification: np.ndarray) -> np.ndarray:
    """
    Create a simplified severity map from classification.
    
    Parameters:
    -----------
    classification : np.ndarray
        Classification array
        
    Returns:
    --------
    np.ndarray
        Severity map with values 0-3:
        0: No change, 1: Slight change, 2: Moderate change, 3: Severe change
    """
    severity = np.abs(classification)
    return severity


def identify_critical_areas(classification: np.ndarray,
                           severity_threshold: int = 2) -> np.ndarray:
    """
    Identify areas requiring attention based on change severity.
    
    Parameters:
    -----------
    classification : np.ndarray
        Classification array
    severity_threshold : int
        Minimum severity level to be considered critical (1-3)
        
    Returns:
    --------
    np.ndarray
        Boolean array where True indicates critical areas
    """
    severity = create_change_severity_map(classification)
    return severity >= severity_threshold


def classify_degradation_risk(ndvi_before: np.ndarray,
                              ndvi_change: np.ndarray) -> np.ndarray:
    """
    Assess degradation risk based on current health and change trend.
    
    Parameters:
    -----------
    ndvi_before : np.ndarray
        NDVI values from earlier period
    ndvi_change : np.ndarray
        NDVI change values
        
    Returns:
    --------
    np.ndarray
        Risk classification:
        0: Low risk, 1: Moderate risk, 2: High risk, 3: Critical risk
    """
    health = classify_ndvi_health(ndvi_before)
    change_class = classify_change_magnitude(ndvi_change)
    
    risk = np.zeros_like(ndvi_before, dtype=np.int8)
    
    # Critical risk: Already degraded and declining
    risk[(health <= VegetationHealthClass.STRESSED) & (change_class < 0)] = 3
    
    # High risk: Moderate health but severe decline
    risk[(health == VegetationHealthClass.MODERATE) & (change_class <= -2)] = 2
    
    # Moderate risk: Healthy but declining, or stressed but stable
    risk[(health >= VegetationHealthClass.HEALTHY) & (change_class <= -1)] = 1
    risk[(health == VegetationHealthClass.STRESSED) & (change_class == 0)] = 1
    
    return risk
