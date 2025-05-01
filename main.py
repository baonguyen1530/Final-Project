# Import pandas for data manipulation and analysis
import pandas as pd
# Import numpy for numerical computations
import numpy as np
# Import Counter from collections for counting occurrences of elements
from collections import Counter
# Import preprocessing tools from sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Import model selection tools for splitting data and cross-validation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
# Import KNN classifier
from sklearn.neighbors import KNeighborsClassifier
# Import Support Vector Machine classifier
from sklearn.svm import SVC
# Import various ensemble classifiers
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
# Import Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
# Import Neural Network classifier
from sklearn.neural_network import MLPClassifier
# Import Bagging classifier
from sklearn.ensemble import BaggingClassifier
# Import feature selection tool
from sklearn.feature_selection import SelectFromModel
# Import metrics for model evaluation
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the data
# Read protein sequences data with custom column names
g_data = pd.read_csv("g_data.csv", header=None, names=['Class', 'Fold', 'ProteinID', 'Sequence'])
# Read occurrence features data
occur_data = pd.read_csv("occur.csv")
# Read physicochemical properties data, skipping the first row
attributes_data = pd.read_csv("attributes.csv", skiprows=1)

# Define the standard amino acid alphabet
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
# Initialize dictionary to store physicochemical properties
attributes_dict = {}

# Process each physicochemical property (using first 9 properties)
for i in range(1, 10):
    # Get property name from the first column
    property_name = attributes_data.iloc[i-1, 1]
    # Initialize dictionary for this property
    property_values = {}
    # For each amino acid, store its property value
    for j, aa in enumerate(amino_acids):
        property_values[aa] = float(attributes_data.iloc[i-1, j+2])
    # Store the property values in the main dictionary
    attributes_dict[property_name] = property_values

# Step 2: Define feature extraction functions
def extract_sequence(sequence):
    """Extract the protein sequence from the input format."""
    # Return the sequence as is since it's already in the correct format
    return sequence

def amino_acid_composition(sequence):
    """Calculate amino acid composition (frequencies) in the sequence."""
    # Extract the sequence
    sequence = extract_sequence(sequence)
    # Count occurrences of each amino acid
    counts = Counter(sequence)
    # Calculate frequency of each amino acid
    composition = {aa: counts.get(aa, 0) / len(sequence) for aa in amino_acids}
    return composition

def dipeptide_composition(sequence):
    """Calculate dipeptide (2-mer) composition in the sequence."""
    # Extract the sequence
    sequence = extract_sequence(sequence)
    # Generate all possible dipeptides
    dipeptides = [aa1+aa2 for aa1 in amino_acids for aa2 in amino_acids]
    
    # Count occurrences of each dipeptide
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
    # Extract the sequence
    sequence = extract_sequence(sequence)
    # Initialize dictionary for properties
    properties = {}
    
    # Calculate average value for each property
    for prop_name, prop_values in attributes_dict.items():
        avg_value = 0
        count = 0
        # Sum up property values for each amino acid
        for aa in sequence:
            if aa in prop_values:
                avg_value += prop_values[aa]
                count += 1
        # Calculate average if we have valid amino acids
        if count > 0:
            avg_value /= count
        properties[f"avg_{prop_name}"] = avg_value
    
    return properties

# Apply feature extraction to all sequences
# Calculate amino acid composition for each sequence
g_data['aa_features'] = g_data['Sequence'].apply(amino_acid_composition)
# Calculate dipeptide composition for each sequence
g_data['dip_features'] = g_data['Sequence'].apply(dipeptide_composition)
# Calculate physicochemical properties for each sequence
g_data['phys_features'] = g_data['Sequence'].apply(avg_physicochemical_properties)

# Convert extracted features into DataFrames
# Convert amino acid features to DataFrame
aa_features = pd.DataFrame(g_data['aa_features'].tolist())
# Convert dipeptide features to DataFrame
dip_features = pd.DataFrame(g_data['dip_features'].tolist())
# Convert physicochemical features to DataFrame
phys_features = pd.DataFrame(g_data['phys_features'].tolist())

# Step 3: Combine features
# Merge all features into a single DataFrame
combined_data = pd.concat([
    occur_data.iloc[:, 1:],  # Occurrence features
    aa_features,             # Amino acid composition
    phys_features,           # Physicochemical properties
    dip_features             # Dipeptide composition
], axis=1)

# Add labels and encode them numerically
# Initialize label encoder
label_encoder = LabelEncoder()
# Convert fold labels to numerical values
labels = label_encoder.fit_transform(g_data['Fold'])
# Create mapping between numerical labels and original fold names
fold_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
print(f"Label mapping: {fold_mapping}")

# Step 4: Normalize and select features
# Initialize standard scaler
scaler = StandardScaler()
# Normalize features
normalized_features = scaler.fit_transform(combined_data)

# Feature selection using Random Forest
print("Performing feature selection...")
# Initialize feature selector with Random Forest
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42), 
    threshold="median"
)
# Fit selector and transform features
selector.fit(normalized_features, labels)
selected_features = selector.transform(normalized_features)
print(f"Selected {selected_features.shape[1]} out of {normalized_features.shape[1]} features")

# Step 5: Split the data into training and testing sets
# First split: create independent test set
X_train_main, X_test_independent, y_train_main, y_test_independent = train_test_split(
    selected_features,        # Use selected features
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

# Step 6: Model Training and Evaluation

# K-Nearest Neighbors (KNN) with hyperparameter tuning
print("\nTuning KNN hyperparameters...")
# Define parameter grid for KNN
knn_params = {
    'n_neighbors': [3, 5, 7, 9],  # Different numbers of neighbors to try
    'weights': ['uniform', 'distance'],  # Weighting schemes
    'p': [1, 2]  # Distance metrics (1=Manhattan, 2=Euclidean)
}
# Initialize GridSearchCV for KNN
knn = GridSearchCV(
    KNeighborsClassifier(), 
    knn_params, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
# Train KNN model
knn.fit(X_train, y_train)
print(f"Best KNN parameters: {knn.best_params_}")
# Make predictions
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", round(accuracy_score(y_test, y_pred_knn), 2))
print(classification_report(y_test, y_pred_knn, zero_division=0))

# Support Vector Machine (SVM) with hyperparameter tuning
print("\nTuning SVM hyperparameters...")
# Define parameter grid for SVM
svm_params = {
    'C': [10],  # Regularization parameter
    'kernel': ['rbf'],  # Radial basis function kernel
    'gamma': ['scale']  # Kernel coefficient
}
# Initialize GridSearchCV for SVM
svm = GridSearchCV(
    SVC(probability=True), 
    svm_params, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
# Train SVM model
svm.fit(X_train, y_train)
print(f"Best SVM parameters: {svm.best_params_}")
# Make predictions
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", round(accuracy_score(y_test, y_pred_svm), 2))
print(classification_report(y_test, y_pred_svm, zero_division=0))

# Random Forest with hyperparameter tuning
print("\nTuning Random Forest hyperparameters...")
# Define parameter grid for Random Forest
rf_params = {
    'n_estimators': [100],  # Number of trees
    'max_depth': [None],  # Maximum depth of trees
    'min_samples_split': [2],  # Minimum samples required to split
    'min_samples_leaf': [1]  # Minimum samples required at leaf node
}
# Initialize GridSearchCV for Random Forest
rf = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    rf_params, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy'
)
# Train Random Forest model
rf.fit(X_train, y_train)
print(f"Best Random Forest parameters: {rf.best_params_}")
# Make predictions
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", round(accuracy_score(y_test, y_pred_rf), 2))
print(classification_report(y_test, y_pred_rf, zero_division=0))

# Naïve Bayes training
print("\nTraining Naive Bayes model...")
# Initialize Naive Bayes classifier
nb = GaussianNB()
# Train Naive Bayes model
nb.fit(X_train, y_train)
# Make predictions
y_pred_nb = nb.predict(X_test)
print("Naïve Bayes Accuracy:", round(accuracy_score(y_test, y_pred_nb), 2))
print(classification_report(y_test, y_pred_nb, zero_division=0))

# Artificial Neural Network (ANN) with hyperparameter tuning
print("\nTuning Neural Network hyperparameters...")
# Define parameter grid for ANN
ann_params = {
    'hidden_layer_sizes': [(100, 50)],  # Network architecture
    'activation': ['relu'],  # Activation function
    'alpha': [0.0001],  # L2 penalty
    'learning_rate': ['adaptive'],  # Learning rate schedule
    'max_iter': [1000]  # Maximum iterations
}
# Initialize GridSearchCV for ANN
ann = GridSearchCV(
    MLPClassifier(random_state=42),
    ann_params,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='accuracy'
)
# Train ANN model
ann.fit(X_train, y_train)
print(f"Best Neural Network parameters: {ann.best_params_}")
# Make predictions
y_pred_ann = ann.predict(X_test)
print("ANN Accuracy:", round(accuracy_score(y_test, y_pred_ann), 2))
print(classification_report(y_test, y_pred_ann, zero_division=0))

# Bagging Classifier
# Initialize Bagging classifier with KNN as base estimator
bagging = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50, random_state=42)
# Train Bagging model
bagging.fit(X_train, y_train)
# Make predictions
y_pred_bagging = bagging.predict(X_test)
print("Bagging Accuracy:", round(accuracy_score(y_test, y_pred_bagging), 2))
print(classification_report(y_test, y_pred_bagging, zero_division=0))

# AdaBoost Classifier
print("\nTraining AdaBoost Classifier...")
# Initialize AdaBoost classifier with Random Forest as base estimator
ada = AdaBoostClassifier(
    estimator=RandomForestClassifier(max_depth=3, random_state=42),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
# Train AdaBoost model
ada.fit(X_train, y_train)
# Make predictions
y_pred_ada = ada.predict(X_test)
print("AdaBoost Accuracy:", round(accuracy_score(y_test, y_pred_ada), 2))
print(classification_report(y_test, y_pred_ada, zero_division=0))

# Stacking Classifier
print("\nTraining Stacking Classifier...")
# Define base estimators for stacking
estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, kernel='rbf', random_state=42))
]
# Initialize Stacking classifier
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    cv=5,
    stack_method='predict_proba'
)
# Train Stacking model
stacking.fit(X_train, y_train)
# Make predictions
y_pred_stacking = stacking.predict(X_test)
print("Stacking Classifier Accuracy:", round(accuracy_score(y_test, y_pred_stacking), 2))
print(classification_report(y_test, y_pred_stacking, zero_division=0))

# Voting Classifier
print("\nTraining Voting Classifier...")
# Initialize Voting classifier with multiple base estimators
voting = VotingClassifier(
    estimators=[
        ('knn', knn.best_estimator_),
        ('rf', rf.best_estimator_),
        ('svm', svm.best_estimator_),
        ('nb', GaussianNB()),
        ('ann', ann.best_estimator_),
        ('ada', ada)
    ],
    voting='soft'  # Use probability-based voting
)
# Train Voting model
voting.fit(X_train, y_train)
# Make predictions
y_pred_voting = voting.predict(X_test)
print("Voting Classifier Accuracy:", round(accuracy_score(y_test, y_pred_voting), 2))
print(classification_report(y_test, y_pred_voting, zero_division=0))

# Step 7: Cross-validation
print("\nPerforming cross-validation with stratified k-fold...")
# Initialize stratified k-fold cross-validator
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation for top models
print("\nRunning cross-validation for top models only to save time...")
# Cross-validate Random Forest
cv_scores_rf = cross_val_score(rf, selected_features, labels, cv=cv, scoring='balanced_accuracy')
print("Random Forest Cross-Validation Balanced Accuracy:", round(cv_scores_rf.mean(), 2))

# Cross-validate SVM
cv_scores_svm = cross_val_score(svm, selected_features, labels, cv=cv, scoring='balanced_accuracy')
print("SVM Cross-Validation Balanced Accuracy:", round(cv_scores_svm.mean(), 2))

# Cross-validate Voting classifier
cv_scores_voting = cross_val_score(voting, selected_features, labels, cv=cv, scoring='balanced_accuracy')
print("Voting Cross-Validation Balanced Accuracy:", round(cv_scores_voting.mean(), 2))

# Step 8: Independent Test Set Evaluation
# Define function to evaluate models on independent test set
def evaluate_model(model, model_name):
    # Make predictions on independent test set
    y_pred = model.predict(X_test_independent)
    # Calculate accuracy
    accuracy = accuracy_score(y_test_independent, y_pred)
    # Generate classification report
    report = classification_report(y_test_independent, y_pred, zero_division=0)
    
    print(f"\n{model_name} Performance on Independent Test Set:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Detailed Classification Report:")
    print(report)
    
    return accuracy

# Evaluate each model on independent test set
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

# Step 9: Save processed data
# Combine normalized features and labels
processed_data = pd.concat([pd.DataFrame(normalized_features), pd.Series(labels, name='Label')], axis=1)
# Save to CSV file
processed_data.to_csv("Data/processed_data.csv", index=False)

# Step 10: Print final summary
print("\n=== Model Accuracy Summary ===")
# Create dictionary of model accuracies
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

# Print accuracy improvement summary
print("\n=== Accuracy Improvement Summary ===")
print("Original KNN Accuracy: 0.70 → New KNN Accuracy: " + str(model_accuracies["KNN"]))
print("Original SVM Accuracy: 0.67 → New SVM Accuracy: " + str(model_accuracies["SVM"]))
print("Original Random Forest Accuracy: 0.71 → New Random Forest Accuracy: " + str(model_accuracies["Random Forest"]))
print("Original Naive Bayes Accuracy: 0.64 → New Naive Bayes Accuracy: " + str(model_accuracies["Naive Bayes"]))
print("Original ANN Accuracy: 0.68 → New ANN Accuracy: " + str(model_accuracies["ANN"]))
print("Original Bagging Accuracy: 0.68 → New Bagging Accuracy: " + str(model_accuracies["Bagging"]))

# Print best model and average improvement
print("\nBest Model: " + sorted_models[0][0] + " with accuracy " + str(sorted_models[0][1]))
print("Average accuracy improvement: " + str(round(((model_accuracies["KNN"] + model_accuracies["SVM"] + model_accuracies["Random Forest"] + model_accuracies["Naive Bayes"] + model_accuracies["ANN"] + model_accuracies["Bagging"])/6 - 0.68), 2)))
