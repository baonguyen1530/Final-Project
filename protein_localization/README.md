# Protein Subcellular Localization Prediction

This project implements a machine learning pipeline for predicting protein subcellular localization in Gram-positive bacteria using various classification methods.

## Project Structure
```
protein_localization/
├── data/                     # Data directory
├── src/                      # Source code
│   ├── feature_extraction.py # Feature extraction utilities
│   ├── data_processing.py    # Data preprocessing
│   ├── models.py            # ML models implementation
│   ├── evaluation.py        # Model evaluation utilities
│   └── main.py             # Main pipeline script
├── notebooks/               # Jupyter notebooks for analysis
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your protein sequence data in the `data/` directory
2. Run the main pipeline:
```bash
python src/main.py
```

## Features
- Feature extraction from protein sequences
- Multiple classification methods:
  - K-Nearest Neighbor (KNN)
  - Support Vector Machine (SVM)
  - Naïve Bayes
  - Artificial Neural Network (ANN)
  - Random Forest
  - Bagging
- Model evaluation using:
  - Independent test set
  - k-fold cross validation
  - Various accuracy measurements

## Results
The results and analysis will be available in the notebooks directory after running the pipeline. 