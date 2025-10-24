# Face vs Logo Detector
Python script that detects whether an image contains a human face or a logo using OpenCV and Deep Learning models.

## Features
- Supports Haar Cascades, LBP, and DNN-based detection.
- Handles multiple image formats (JPEG, PNG, HEIC, AVIF).
- Saves annotated images with detected faces.

## Requirements
- Python 3.9+
- Libraries: OpenCV, NumPy, Pillow, pandas, SQLite3, PyTorch (optional for DNN)

## Usage
```bash
python3 face_logo_detector.py --data path/to/database.db --avatars_dir path/to/images/
