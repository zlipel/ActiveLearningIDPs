import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Function to convert and normalize the features
def convert_and_normalize_features(df_inp):
    """Normalize raw feature DataFrame in-place.

    Parameters
    ----------
    df_inp : pandas.DataFrame
        Raw features where counts are absolute and lengths unscaled.

    Returns
    -------
    pandas.DataFrame
        Normalized feature set where compositional features are expressed as
        fractions and other descriptors are standardized.
    """

    # Step 1: Convert features to proper form
    df = df_inp.copy()
    for col in df.columns[:20]:
        df[col] = df[col] / df['length']

    # Convert total charges to fractions
    df['beads(+)'] = df['beads(+)'] / df['length']
    df['beads(-)'] = df['beads(-)'] / df['length']

    df['|net charge|'] = df['|net charge|'] / df['length']

    df['sum lambda'] = df['sum lambda'] / df['length']

    df['mol wt'] = df['mol wt'] / df['length']

    # Step 2: Normalize the features
    features_to_standard_normalize = [
        'beads(+)', 'beads(-)', 'sum lambda', 'mol wt', 'SHD', 'SCD', '|net charge|'
    ]
    for feature in features_to_standard_normalize:
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean) / std

    # Min-max normalization for sequence length L
    min_L = df['length'].min()
    max_L = df['length'].max()
    df['length'] = (df['length'] - min_L) / (max_L - min_L)

    # Normalize Shannon entropy S by maximum
    max_S = df['shan ent'].max()
    df['shan ent'] = df['shan ent'] / max_S - 1

    return df

# Function to load and prepare the dataset
def load_dataset(features_file, labels_file, label_column='B2', model=None):
    """
    Load and prepare the dataset for training.
    
    Parameters:
    features_file (str): Path to the CSV file containing the feature data.
    labels_file (str): Path to the CSV file containing the labels.
    label_column (str): The name of the label column to extract from the labels file.
    
    Returns:
    Dataset: A PyTorch dataset with the features and labels ready for training.
    """
    # Load and normalize features
    features_df = pd.read_csv(features_file)
    features_df = convert_and_normalize_features(features_df)
    
    # Load labels
    labels_df = pd.read_csv(labels_file)

    # make sure we have no nan values for training...

    labels_nonan = labels_df.dropna(subset=label_column)
    feats_nonan = features_df[features_df.index.isin(labels_nonan.index)]
    
    # Extract labels for the specified property
    #if label_column == 'diff':
    #    labels = np.log(labels_nonan[label_column].values+1e-8)
    #else:
    #    labels = labels_nonan[label_column].values
    labels = labels_nonan[label_column].values
    
    # Convert features to tensor
    features = feats_nonan.values
    
    # Convert labels to tensor
    #labels_tensor = torch.tensor(labels).float()
    # if scaler:
    #     scaler = StandardScaler()
    #     scaler.fit(labels_tensor.reshape(-1,1))
    #     labels_fit = scaler.transform(labels_tensor.reshape(-1,1))
    
    #Create dataset
    if model == 'dnn':
        dataset = ProteinDataset(features, labels)
        return dataset
    else:
        
        return features, labels

class ProteinDataset(Dataset):
    """Dataset wrapper for regression tasks on protein features."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Retrieve the feature vector and label for a given index."""
        return self.features[idx], self.labels[idx]

def load_classification_dataset(features_file, labels_file, label_column='psp'):
    """
    Load and prepare the dataset for classification.

    Parameters:
    features_file (str): Path to the CSV file containing the feature data.
    labels_file (str): Path to the CSV file containing the labels.
    label_column (str): The name of the label column to extract from the labels file.

    Returns:
    Dataset: A PyTorch dataset with the features and labels ready for training.
    """
    # Load and normalize features
    features_df = pd.read_csv(features_file)
    features_df = convert_and_normalize_features(features_df)
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    
    # Extract classification labels for the specified property
    labels = labels_df[label_column].values
    
    # Convert features to tensor
    features_tensor = torch.tensor(features_df.values).float()
    
    # Convert labels to tensor (use long for classification)
    labels_tensor = torch.tensor(labels).long()
    
    # Create dataset
    dataset = ClassificationDataset(features_tensor, labels_tensor)
    
    return dataset

class ClassificationDataset(Dataset):
    """Dataset wrapper for classification tasks on protein features."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Retrieve the feature vector and label for a given index."""
        return self.features[idx], self.labels[idx]
