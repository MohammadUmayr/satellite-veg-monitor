# 🛰️ Satellite Vegetation Monitor

Python library for satellite-based vegetation monitoring and temporal change detection using Landsat and Sentinel-2 imagery.

## 📋 Features

- **Data Loading**: Load and preprocess Landsat 8/9 and Sentinel-2 imagery
- **Atmospheric Correction**: Apply DOS (Dark Object Subtraction) and TOA corrections
- **Spectral Indices**: Calculate NDVI, EVI, SAVI, NDWI, NBR
- **Change Detection**: Analyze vegetation changes between two time periods
- **Change Classification**: Classify change severity with multiple thresholds

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/satellite-veg-monitor.git
cd satellite-veg-monitor

# Install dependencies
pip install -r requirements.txt
```

## 📖 Quick Start

### Single Date Analysis
```python
from satellite_veg_monitor import data_loader, index_calculator

# Load satellite image
bands, metadata = data_loader.load_satellite_image(
    'path/to/satellite/image/',
    satellite_type='landsat8',
    apply_correction=True
)

# Calculate indices
indices = index_calculator.calculate_all_indices(bands)

# Save NDVI
index_calculator.save_index(indices['ndvi'], 'ndvi_output.tif', metadata)
```

### Temporal Change Detection
```python
from satellite_veg_monitor import change_detector, change_classifier

# Load two time periods
indices_2020 = index_calculator.calculate_all_indices(bands_2020)
indices_2024 = index_calculator.calculate_all_indices(bands_2024)

# Detect changes
changes = change_detector.analyze_all_indices_change(indices_2020, indices_2024)

# Classify changes
classifications = change_classifier.classify_all_changes(changes)
```

## 📁 Project Structure
```
satellite-veg-monitor/
├── satellite_veg_monitor/     # Main package
│   ├── data_loader.py         # Image loading & preprocessing
│   ├── index_calculator.py    # Spectral index calculations
│   ├── change_detector.py     # Temporal change analysis
│   └── change_classifier.py   # Change classification
├── tests/                     # Unit tests
├── examples/                  # Usage examples
├── data/                      # Sample data
└── docs/                      # Documentation
```

## 🧪 Testing
```bash
python -m pytest tests/
```

## 👥 Contributors

- Ola Elwasila - Module 1 & 2
- Mohmmad Umayr Romshoo - Module 3 & 4

## 📄 License

MIT License

## 🙏 Acknowledgments

This project was developed as part of the Geospatial Processing course at Politecnico di Milano.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
