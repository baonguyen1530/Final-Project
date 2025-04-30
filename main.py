import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report





# Step 1: Load the data
g_data = pd.read_csv("g_data.csv", header=None, names=['Class', 'Fold', 'ProteinID', 'Sequence'])  # Protein sequences
occur_data = pd.read_csv("occur.csv")  # Occurrence features
attributes_data = pd.read_csv("attributes.csv", skiprows=1)  # Physicochemical properties


# Process the attributes data to create a dictionary for easy access
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
attributes_dict = {}


# Process each physicochemical property
for i in range(1, 10):  # Using first 9 properties for simplicity, can be extended
    property_name = attributes_data.iloc[i-1, 1]
    property_values = {}
    for j, aa in enumerate(amino_acids):
        property_values[aa] = float(attributes_data.iloc[i-1, j+2])
    attributes_dict[property_name] = property_values



# Step 2: Extract features from protein sequences
def extract_sequence(sequence):
    """Extract the protein sequence from the input format."""
    # No need to split the sequence now since it's already in its own column
    return sequence

def amino_acid_composition(sequence):
    """Calculate amino acid composition (frequencies) in the sequence."""
    sequence = extract_sequence(sequence)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counts = Counter(sequence)
    composition = {aa: counts.get(aa, 0) / len(sequence) for aa in amino_acids}
    return composition

def dipeptide_composition(sequence):
    """Calculate dipeptide (2-mer) composition in the sequence."""
    sequence = extract_sequence(sequence)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1+aa2 for aa1 in amino_acids for aa2 in amino_acids]
    
    # Count dipeptides
    dip_counts = {}
    for i in range(len(sequence)-1):
        dipeptide = sequence[i:i+2]
        if all(aa in amino_acids for aa in dipeptide):
            dip_counts[dipeptide] = dip_counts.get(dipeptide, 0) + 1
    
    # Normalize by total number of dipeptides
    total_dipeptides = max(1, len(sequence)-1)  # Avoid division by zero
    dip_composition = {dip: dip_counts.get(dip, 0) / total_dipeptides for dip in dipeptides}
    return dip_composition

def avg_physicochemical_properties(sequence):
    """Calculate average physicochemical properties for the sequence."""
    sequence = extract_sequence(sequence)
    properties = {}
    
    # Calculate average value for each property
    for prop_name, prop_values in attributes_dict.items():
        avg_value = 0
        count = 0
        for aa in sequence:
            if aa in prop_values:
                avg_value += prop_values[aa]
                count += 1
        if count > 0:
            avg_value /= count
        properties[f"avg_{prop_name}"] = avg_value
    
    return properties

# Apply feature extraction to all sequences
g_data['aa_features'] = g_data['Sequence'].apply(amino_acid_composition)
g_data['dip_features'] = g_data['Sequence'].apply(dipeptide_composition)
g_data['phys_features'] = g_data['Sequence'].apply(avg_physicochemical_properties)

# Convert extracted features into DataFrames
aa_features = pd.DataFrame(g_data['aa_features'].tolist())
dip_features = pd.DataFrame(g_data['dip_features'].tolist())
phys_features = pd.DataFrame(g_data['phys_features'].tolist())









# Step 3: Combine features
# Merge occurrence features and all sequence-derived features
combined_data = pd.concat([
    occur_data.iloc[:, 1:],  # Occurrence features
    aa_features,             # Amino acid composition
    phys_features,           # Physicochemical properties
    dip_features             # Dipeptide composition
], axis=1)

# Add labels and encode them numerically
# Extract the fold information
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(g_data['Fold'])  # Now this will get Fold1, Fold2, etc.
fold_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
print(f"Label mapping: {fold_mapping}")










# Step 4: Normalize and select features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(combined_data)

# Note: These train/test split operations have been moved after feature selection
# Feature selection using Random Forest
print("Performing feature selection...")
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42), 
    threshold="median"
)
selector.fit(normalized_features, labels)
selected_features = selector.transform(normalized_features)
print(f"Selected {selected_features.shape[1]} out of {normalized_features.shape[1]} features")

# Step 5: Split the data into training and testing sets (with stratification)
X_train_main, X_test_independent, y_train_main, y_test_independent = train_test_split(
    selected_features,        # Use selected features instead of normalized_features
    labels,
    test_size=0.2,          # 20% for independent testing
    random_state=42,
    stratify=labels         # ensure balanced class distribution
)

# Second split: create validation set from training data
X_train, X_test, y_train, y_test = train_test_split(
    X_train_main,
    y_train_main,
    test_size=0.25,
    random_state=42,
    stratify=y_train_main
)

# Step 6: Hyperparameter tuning and model training

# K-Nearest Neighbors (KNN) with hyperparameter tuning
print("\nTuning KNN hyperparameters...")
knn_params = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # p=1 is Manhattan distance, p=2 is Euclidean
}
knn = GridSearchCV(
    KNeighborsClassifier(), 
    knn_params, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
knn.fit(X_train, y_train)
print(f"Best KNN parameters: {knn.best_params_}")
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", round(accuracy_score(y_test, y_pred_knn), 2))
print(classification_report(y_test, y_pred_knn, zero_division=0))

# Support Vector Machine (SVM) with hyperparameter tuning
print("\nTuning SVM hyperparameters...")
svm_params = {
    'C': [10],
    'kernel': ['rbf'],
    'gamma': ['scale']
}
svm = GridSearchCV(
    SVC(probability=True), 
    svm_params, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
svm.fit(X_train, y_train)
print(f"Best SVM parameters: {svm.best_params_}")
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", round(accuracy_score(y_test, y_pred_svm), 2))
print(classification_report(y_test, y_pred_svm, zero_division=0))

# Random Forest with hyperparameter tuning
print("\nTuning Random Forest hyperparameters...")
rf_params = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}
rf = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    rf_params, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
rf.fit(X_train, y_train)
print(f"Best Random Forest parameters: {rf.best_params_}")
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", round(accuracy_score(y_test, y_pred_rf), 2))
print(classification_report(y_test, y_pred_rf, zero_division=0))

# Naïve Bayes with parameter exploration
print("\nTraining Naive Bayes model...")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naïve Bayes Accuracy:", round(accuracy_score(y_test, y_pred_nb), 2))
print(classification_report(y_test, y_pred_nb, zero_division=0))

# Artificial Neural Network (ANN) with hyperparameter tuning
print("\nTuning Neural Network hyperparameters...")
ann_params = {
    'hidden_layer_sizes': [(100, 50)],
    'activation': ['relu'],
    'alpha': [0.0001],
    'learning_rate': ['adaptive'],
    'max_iter': [1000]
}
ann = GridSearchCV(
    MLPClassifier(random_state=42),
    ann_params,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # Using 3-fold to save time
    scoring='accuracy'
)
ann.fit(X_train, y_train)
print(f"Best Neural Network parameters: {ann.best_params_}")
y_pred_ann = ann.predict(X_test)
print("ANN Accuracy:", round(accuracy_score(y_test, y_pred_ann), 2))
print(classification_report(y_test, y_pred_ann, zero_division=0))

# Bagging Classifier
bagging = BaggingClassifier(estimator = KNeighborsClassifier(), n_estimators = 50, random_state = 42)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
print("Bagging Accuracy:", round(accuracy_score(y_test, y_pred_bagging), 2))
print(classification_report(y_test, y_pred_bagging, zero_division=0))
# AdaBoost Classifier
print("\nTraining AdaBoost Classifier...")
ada = AdaBoostClassifier(
    estimator=RandomForestClassifier(max_depth=3, random_state=42),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("AdaBoost Accuracy:", round(accuracy_score(y_test, y_pred_ada), 2))
print(classification_report(y_test, y_pred_ada, zero_division=0))
# Stacking Classifier
print("\nTraining Stacking Classifier...")
estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, kernel='rbf', random_state=42))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=5,
    stack_method='predict_proba'
)
stacking.fit(X_train, y_train)
y_pred_stacking = stacking.predict(X_test)
print("Stacking Classifier Accuracy:", round(accuracy_score(y_test, y_pred_stacking), 2))
print(classification_report(y_test, y_pred_stacking, zero_division=0))

# Voting Classifier
print("\nTraining Voting Classifier...")
voting = VotingClassifier(
    estimators=[
        ('knn', knn.best_estimator_),
        ('rf', rf.best_estimator_),
        ('svm', svm.best_estimator_),
        ('nb', GaussianNB()),
        ('ann', ann.best_estimator_),
        ('ada', ada)
    ],
    voting='soft'
)
voting.fit(X_train, y_train)
y_pred_voting = voting.predict(X_test)
print("Voting Classifier Accuracy:", round(accuracy_score(y_test, y_pred_voting), 2))
print(classification_report(y_test, y_pred_voting, zero_division=0))








# Step 7: Cross-validation
print("\nPerforming cross-validation with stratified k-fold...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning cross-validation for top models only to save time...")
cv_scores_rf = cross_val_score(rf, selected_features, labels, cv=cv, scoring='balanced_accuracy')
print("Random Forest Cross-Validation Balanced Accuracy:", round(cv_scores_rf.mean(), 2))

cv_scores_svm = cross_val_score(svm, selected_features, labels, cv=cv, scoring='balanced_accuracy')
print("SVM Cross-Validation Balanced Accuracy:", round(cv_scores_svm.mean(), 2))

cv_scores_voting = cross_val_score(voting, selected_features, labels, cv=cv, scoring='balanced_accuracy')
print("Voting Cross-Validation Balanced Accuracy:", round(cv_scores_voting.mean(), 2))




# NEW STEP
# Evaluate all models on the independent test set
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test_independent)
    accuracy = accuracy_score(y_test_independent, y_pred)
    report = classification_report(y_test_independent, y_pred, zero_division=0)
    
    print(f"\n{model_name} Performance on Independent Test Set:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Detailed Classification Report:")
    print(report)
    
    return accuracy

# Evaluate each model
independent_results = {
    'KNN': evaluate_model(knn, "K-Nearest Neighbors"),
    'SVM': evaluate_model(svm, "Support Vector Machine"),
    'RF': evaluate_model(rf, "Random Forest"),
    'NB': evaluate_model(nb, "Naive Bayes"),
    'ANN': evaluate_model(ann, "Artificial Neural Network"),
    'Bagging': evaluate_model(bagging, "Bagging")
}

# Compare model performances
print("\nModel Performance Comparison on Independent Test Set:")
for model_name, accuracy in independent_results.items():
    print(f"{model_name}: {accuracy:.2f}")









# Step 8: Save processed data (optional)
processed_data = pd.concat([pd.DataFrame(normalized_features), pd.Series(labels, name='Label')], axis=1)
processed_data.to_csv("Data/processed_data.csv", index=False)


# Print summary of best models (based on test accuracy)
print("\n=== Model Accuracy Summary ===")
model_accuracies = {
    "KNN": round(accuracy_score(y_test, y_pred_knn), 2),
    "SVM": round(accuracy_score(y_test, y_pred_svm), 2),
    "Random Forest": round(accuracy_score(y_test, y_pred_rf), 2),
    "Naive Bayes": round(accuracy_score(y_test, y_pred_nb), 2),
    "ANN": round(accuracy_score(y_test, y_pred_ann), 2),
    "Bagging": round(accuracy_score(y_test, y_pred_bagging), 2),
    "AdaBoost": round(accuracy_score(y_test, y_pred_ada), 2),
    "Stacking": round(accuracy_score(y_test, y_pred_stacking), 2),
    "Voting": round(accuracy_score(y_test, y_pred_voting), 2)
}


# Sort models by accuracy
sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)


# Print models in order of performance
for model_name, accuracy in sorted_models:
    print(f"{model_name} Accuracy: {accuracy}")

print("\n=== Accuracy Improvement Summary ===")
print("Original KNN Accuracy: 0.70 → New KNN Accuracy: " + str(model_accuracies["KNN"]))
print("Original SVM Accuracy: 0.67 → New SVM Accuracy: " + str(model_accuracies["SVM"]))
print("Original Random Forest Accuracy: 0.71 → New Random Forest Accuracy: " + str(model_accuracies["Random Forest"]))
print("Original Naive Bayes Accuracy: 0.64 → New Naive Bayes Accuracy: " + str(model_accuracies["Naive Bayes"]))
print("Original ANN Accuracy: 0.68 → New ANN Accuracy: " + str(model_accuracies["ANN"]))
print("Original Bagging Accuracy: 0.68 → New Bagging Accuracy: " + str(model_accuracies["Bagging"]))

print("\nBest Model: " + sorted_models[0][0] + " with accuracy " + str(sorted_models[0][1]))
print("Average accuracy improvement: " + str(round(((model_accuracies["KNN"] + model_accuracies["SVM"] + model_accuracies["Random Forest"] + model_accuracies["Naive Bayes"] + model_accuracies["ANN"] + model_accuracies["Bagging"])/6 - 0.68), 2)))