# Protein Subcellular Localization - Machine Learning Project

This project uses machine learning to predict the subcellular localization of Gram-positive bacterial proteins.

## Setup and Usage

### Using GitHub Codespaces
1. Click the "Code" button on the GitHub repository
2. Select "Open with Codespaces"
3. Click "New codespace"
4. The environment will automatically install all dependencies

### Running the Code
- To run the main script: `python main.py`
- To use the Jupyter notebook: `jupyter notebook main.ipynb`

## Dataset
The project uses three main data files:
- `g_data.csv`: Protein sequences data
- `occur.csv`: Occurrence features
- `attributes.csv`: Physicochemical properties

## Features
- Amino acid composition
- Dipeptide composition
- Physicochemical properties
- Occurrence features

## Models
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes
- Artificial Neural Network (ANN)
- Ensemble methods (Bagging, AdaBoost, Stacking, Voting)