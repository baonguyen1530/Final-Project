import os
import pandas as pd
import numpy as np
from data_processing import DataProcessor
from models import ModelFactory, ModelEvaluator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_all_features():
    """Load and combine all available feature sets."""
    # Define data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    comp_path = os.path.join(data_dir, 'comp.csv')
    occur_path = os.path.join(data_dir, 'occur.csv')
    sequence_path = os.path.join(data_dir, 'n-data.csv')
    
    # Load composition features
    comp_df = pd.read_csv(comp_path)
    
    # Load occurrence features
    occur_df = pd.read_csv(occur_path)
    
    # Load sequence data
    seq_df = pd.read_csv(sequence_path, header=None, names=['id', 'fold', 'protein_id', 'sequence'])
    
    # Combine features
    features = pd.merge(comp_df, occur_df.drop('Fold', axis=1), 
                       left_index=True, right_index=True, 
                       suffixes=('_comp', '_occur'))
    
    return features, seq_df

def main():
    # Load data
    print("Loading data...")
    features, sequences = load_all_features()
    
    # Split data into training and testing sets
    X = features.drop('Fold', axis=1).values
    y = features['Fold'].values
    
    # Encode labels for neural network
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'KNN': ModelFactory.create_model('knn'),
        'SVM': ModelFactory.create_model('svm'),
        'Naive Bayes': ModelFactory.create_model('naive_bayes'),
        'Random Forest': ModelFactory.create_model('random_forest'),
        'Bagging': ModelFactory.create_model('bagging'),
        'Neural Network': ModelFactory.create_model('mlp')
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train and evaluate
        eval_results = ModelEvaluator.evaluate_model(model, X_train, X_test, y_train, y_test)
        cv_results = ModelEvaluator.cross_validate(model, X, y)
        
        results[name] = {
            'test_accuracy': eval_results['accuracy'],
            'cv_mean': cv_results['mean_cv_score'],
            'cv_std': cv_results['std_cv_score']
        }
        
        print(f"{name} Results:")
        print(f"Test Accuracy: {results[name]['test_accuracy']:.4f}")
        print(f"CV Score: {results[name]['cv_mean']:.4f} (+/- {results[name]['cv_std']*2:.4f})")
    
    # Create ANN model
    print("\nTraining Deep Learning Model...")
    input_shape = X.shape[1]
    num_classes = len(np.unique(y))
    ann_model = ModelFactory.create_ann(input_shape, num_classes)
    
    # Train and evaluate ANN with encoded labels
    X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    ann_results = ModelEvaluator.evaluate_ann(ann_model, X_train_ann, X_test_ann, y_train_ann, y_test_ann)
    print("\nANN Results:")
    print(f"Test Accuracy: {ann_results['accuracy']:.4f}")
    print(f"Test Loss: {ann_results['loss']:.4f}")

if __name__ == "__main__":
    main() 