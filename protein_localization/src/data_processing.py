import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from feature_extraction import ProteinFeatureExtractor

class DataProcessor:
    def __init__(self):
        self.feature_extractor = ProteinFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_fasta_data(self, fasta_file, labels_file):
        """Load protein sequences and their labels."""
        # Read sequences
        sequences = {}
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = str(record.seq)
            
        # Read labels
        labels_df = pd.read_csv(labels_file)
        
        # Match sequences with labels
        data = []
        for idx, row in labels_df.iterrows():
            if row['protein_id'] in sequences:
                data.append({
                    'protein_id': row['protein_id'],
                    'sequence': sequences[row['protein_id']],
                    'location': row['location']
                })
        
        return pd.DataFrame(data)
    
    def prepare_data(self, data_df, test_size=0.2, random_state=42):
        """Prepare data for machine learning."""
        # Extract features
        X = np.array([
            self.feature_extractor.extract_all_features(seq)
            for seq in data_df['sequence']
        ])
        
        # Encode labels
        y = self.label_encoder.fit_transform(data_df['location'])
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        """Get names of all features."""
        return self.feature_extractor.get_feature_names()
    
    def get_location_names(self):
        """Get names of subcellular locations."""
        return self.label_encoder.classes_
    
    def predict_location(self, sequence):
        """Predict location for a new sequence."""
        features = self.feature_extractor.extract_all_features(sequence)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        return features_scaled 