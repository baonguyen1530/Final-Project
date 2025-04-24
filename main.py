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

# this checks whether the dataset has only one column
# data.shape returns a tuple(rows,colums)
# safeguard to ensure the data is properly formatted
if data.shape[1] == 1:
    data = data[0].str.split(',', expand = True)

# this extracts a subset of the DataFrame
# 1: selects all rows starting from the second row (index 1) to the end
# 1: selects all column (index 1) to the end
x = data.iloc[1:,1:]

# y_initial extract all the rows and only the first column
y_initial = data.iloc[1:,0]

#LabelEncoder in scikit-learn is a tool used to convert categorical labels into numerical values
label_encoder_class = LabelEncoder()

y = label_encoder_class.fit_transform(y_initial)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

knn_results = []

kvalue = [1,5,10,25]
for k in kvalue:
    knn = KNeighborsClassifier(k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    knn_csv = KNeighborsClassifier(n_neighbors = k)
    cv_accuracy = cross_val_score(knn_csv, x, y, cv = 5, scoring = 'accuracy')
    cv_accuracy = cv_accuracy.mean()

    mcc = matthews_corrcoef(y_test, y_pred)
    sensitivity, specificity = multi_class_performance(y_test, y_pred)

    knn_results.append({
        'K': k,
        'accuracy': cv_accuracy,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity
    })

for result in knn_results:
    print(f'K: {result['K']}')
    print(f'Accuracy: {result['accuracy']:.3f}')
    print(f'MCC: {result['mcc']:.3f}')
    print(f'Sensitivity: {result['sensitivity']:.3f}')
    print(f'Specificity: {result['specificity']:.3f}')
    print()