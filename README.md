# Seismic Reflector Horizon Mapper

An interactive Streamlit app to detect, track, and visualize reflector horizons 
from seismic-style BMP images. The app overlays a smooth green line along the 
desired contour and denoises the region around it.

## Features
- Upload single or multiple BMP images
- Automatic edge detection and contour tracking
- Interactive parameter controls:
  - Target depth and search margin
  - Continuity window
  - Gradient vs distance weighting
  - Median and Savitzky–Golay smoothing
  - Line thickness
- Handles discontinuities with interpolation
- Batch processing with ZIP download
- Optional debug views for edges and masks

## Usage
```bash
streamlit run edge_map.py
