"""
Satellite Vegetation Monitor
Python library for satellite-based vegetation monitoring and temporal change detection.

Modules:
- data_loader: Image loading & preprocessing
- index_calculator: Spectral index calculations
- change_detector: Temporal change analysis (Module 3 - Mohmmad Umayr Romshoo)
- change_classifier: Change classification (Module 4 - Mohmmad Umayr Romshoo)
"""

__version__ = "0.1.0"
__author__ = "Ola Elwasila, Mohmmad Umayr Romshoo"

# Import main modules when they're available
try:
    from . import change_detector
    from . import change_classifier
except ImportError:
    pass

__all__ = ['change_detector', 'change_classifier']
