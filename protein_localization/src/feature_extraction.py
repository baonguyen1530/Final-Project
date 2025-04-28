from Bio import SeqIO
import numpy as np
from collections import Counter
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis

class ProteinFeatureExtractor:
    def __init__(self):
        self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        self.dipeptides = [a1 + a2 for a1 in self.amino_acids for a2 in self.amino_acids]
        
        # Define physicochemical properties
        self.hydrophobic = set('AILMFWYV')
        self.hydrophilic = set('RHKDESTNQ')
        self.charged = set('RHKDE')
        self.polar = set('RHKDESTNQY')
        self.aliphatic = set('ILV')
        self.aromatic = set('FWY')
    
    def extract_composition(self, sequence):
        """Calculate amino acid composition features."""
        sequence = sequence.upper()
        counts = Counter(sequence)
        composition = np.zeros(20)
        for i, aa in enumerate(self.amino_acids):
            composition[i] = counts.get(aa, 0) / len(sequence)
        return composition
    
    def extract_dipeptide_composition(self, sequence):
        """Calculate dipeptide composition features."""
        sequence = sequence.upper()
        pairs = [sequence[i:i+2] for i in range(len(sequence)-1)]
        counts = Counter(pairs)
        composition = np.zeros(400)
        for i, dipep in enumerate(self.dipeptides):
            composition[i] = counts.get(dipep, 0) / (len(sequence)-1)
        return composition
    
    def calculate_molecular_weight(self, sequence):
        """Calculate molecular weight of the protein."""
        try:
            protein = ProteinAnalysis(sequence)
            return protein.molecular_weight()
        except:
            # Fallback to simple calculation if Bio.SeqUtils fails
            weights = {
                'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
                'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
                'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
                'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
            }
            return sum(weights.get(aa, 0) for aa in sequence.upper())
    
    def calculate_hydrophobicity(self, sequence):
        """Calculate average hydrophobicity using Kyte-Doolittle scale."""
        hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        return np.mean([hydrophobicity.get(aa, 0) for aa in sequence.upper()])
    
    def extract_physicochemical_properties(self, sequence):
        """Extract various physicochemical properties."""
        sequence = sequence.upper()
        length = len(sequence)
        
        # Calculate property percentages
        hydrophobic_percent = sum(aa in self.hydrophobic for aa in sequence) / length
        hydrophilic_percent = sum(aa in self.hydrophilic for aa in sequence) / length
        charged_percent = sum(aa in self.charged for aa in sequence) / length
        polar_percent = sum(aa in self.polar for aa in sequence) / length
        aliphatic_percent = sum(aa in self.aliphatic for aa in sequence) / length
        aromatic_percent = sum(aa in self.aromatic for aa in sequence) / length
        
        try:
            protein = ProteinAnalysis(sequence)
            instability_index = protein.instability_index()
            aromaticity = protein.aromaticity()
            gravy = protein.gravy()  # Grand average of hydropathy
            secondary_structure = protein.secondary_structure_fraction()
            isoelectric_point = protein.isoelectric_point()
        except:
            # Fallback values if Bio.SeqUtils fails
            instability_index = 0
            aromaticity = aromatic_percent
            gravy = self.calculate_hydrophobicity(sequence)
            secondary_structure = (0, 0, 0)
            isoelectric_point = 7.0
            
        return np.array([
            hydrophobic_percent,
            hydrophilic_percent,
            charged_percent,
            polar_percent,
            aliphatic_percent,
            aromatic_percent,
            instability_index,
            aromaticity,
            gravy,
            secondary_structure[0],  # Helix fraction
            secondary_structure[1],  # Turn fraction
            secondary_structure[2],  # Sheet fraction
            isoelectric_point
        ])
    
    def extract_sequence_properties(self, sequence):
        """Extract basic sequence properties."""
        sequence = sequence.upper()
        length = len(sequence)
        
        # Calculate sequence complexity features
        unique_aa = len(set(sequence))
        unique_aa_ratio = unique_aa / length
        
        # Calculate repeats
        repeat_pattern = re.findall(r'(.+?)\1+', sequence)
        repeat_count = len(repeat_pattern)
        
        return np.array([length, unique_aa, unique_aa_ratio, repeat_count])
    
    def extract_all_features(self, sequence):
        """Extract all features for a given protein sequence."""
        features = np.concatenate([
            self.extract_composition(sequence),
            self.extract_dipeptide_composition(sequence),
            [self.calculate_molecular_weight(sequence)],
            [self.calculate_hydrophobicity(sequence)],
            self.extract_physicochemical_properties(sequence),
            self.extract_sequence_properties(sequence)
        ])
        return features
    
    def get_feature_names(self):
        """Get names of all features."""
        names = (
            [f"aa_comp_{aa}" for aa in self.amino_acids] +
            [f"dipep_comp_{dp}" for dp in self.dipeptides] +
            ["molecular_weight", "hydrophobicity"] +
            ["hydrophobic_percent", "hydrophilic_percent", "charged_percent",
             "polar_percent", "aliphatic_percent", "aromatic_percent",
             "instability_index", "aromaticity", "gravy",
             "helix_fraction", "turn_fraction", "sheet_fraction",
             "isoelectric_point"] +
            ["sequence_length", "unique_aa_count", "unique_aa_ratio", "repeat_count"]
        )
        return names 