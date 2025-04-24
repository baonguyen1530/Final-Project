import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

def multi_class_performance(y_true, y_pred):
    """
    Calculate sensitivity and specificity for multi-class classification.

    Parameters:
    - y_true: array-like, true class labels
    - y_pred: array-like, predicted class labels

    Returns:
    - sensitivity: float, average recall across all classes
    - specificity: float, average precision across all classes
    """
    # Generate a classification report as a dictionary
    classification_report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    
    # Calculate sensitivity (average recall across all classes)
    sensitivity = np.mean([
        classification_report_dict[str(class_label)]['recall']
        for class_label in classification_report_dict if class_label.isdigit()
    ])
    
    # Calculate specificity (average precision across all classes)
    specificity = np.mean([
        classification_report_dict[str(class_label)]['precision']
        for class_label in classification_report_dict if class_label.isdigit()
    ])
    
    return sensitivity, specificity

# read the data from comp_occur (1).csv
data = pd.read_csv("Data/comp_occur (1).csv", delimiter = '\t', header = None)

print(data)