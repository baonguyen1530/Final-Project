from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class ModelFactory:
    @staticmethod
    def create_model(model_name, **kwargs):
        """Create a model based on the model name."""
        models = {
            'knn': KNeighborsClassifier(n_neighbors=5, **kwargs),
            'svm': SVC(kernel='rbf', probability=True, **kwargs),
            'naive_bayes': GaussianNB(**kwargs),
            'random_forest': RandomForestClassifier(n_estimators=100, **kwargs),
            'bagging': BaggingClassifier(n_estimators=100, **kwargs),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, **kwargs)
        }
        return models.get(model_name.lower())

    @staticmethod
    def create_ann(input_shape, num_classes):
        """Create a deep learning model using TensorFlow/Keras."""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        """Evaluate a model using various metrics."""
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    @staticmethod
    def cross_validate(model, X, y, cv=5):
        """Perform k-fold cross validation."""
        scores = cross_val_score(model, X, y, cv=cv)
        return {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),
            'cv_scores': scores
        }
    
    @staticmethod
    def evaluate_ann(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=32):
        """Evaluate the ANN model."""
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'history': history.history
        } 