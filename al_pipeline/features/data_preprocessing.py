import pandas as pd
import torch
from torch.utils.data import Dataset

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
    df = df_inp.copy()

    # Convert counts to prevalence for amino acids (first 20 columns)
    length = df['length']
    for col in df.columns[:20]:
        df[col] = df[col] / length

    # Convert global statistics to length-normalized values
    length_scaled_features = ['beads(+)', 'beads(-)', '|net charge|', 'sum lambda', 'mol wt']
    for feature in length_scaled_features:
        df[feature] = df[feature] / length

    # Standard normalization for selected features
    features_to_standard_normalize = length_scaled_features + ['SHD', 'SCD']
    for feature in features_to_standard_normalize:
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean) / std

    # Min-max normalization for sequence length L
    min_length = df['length'].min()
    max_length = df['length'].max()
    df['length'] = (df['length'] - min_length) / (max_length - min_length)

    # Normalize Shannon entropy S by maximum
    max_entropy = df['shan ent'].max()
    df['shan ent'] = df['shan ent'] / max_entropy - 1

    return df


# Function to load and prepare the dataset
def load_dataset(features_file, labels_file, label_column='B2'):
    """Load and prepare the dataset for training.

    Parameters
    ----------
    features_file : str
        Path to the CSV file containing the feature data.
    labels_file : str
        Path to the CSV file containing the labels.
    label_column : str
        The name of the label column to extract from the labels file.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Normalized feature matrix and label vector aligned by index.
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
    # if label_column == 'diff':
    #     labels = np.log(labels_nonan[label_column].values+1e-8)
    # else:
    #     labels = labels_nonan[label_column].values
    labels = labels_nonan[label_column].values

    # Convert features to a NumPy array
    features = feats_nonan.values

    return features, labels


# Function to load and prepare the dataset
def load_classification_dataset(features_file, labels_file, label_column='psp'):
    """Load and prepare the dataset for classification.

    Parameters
    ----------
    features_file : str
        Path to the CSV file containing the feature data.
    labels_file : str
        Path to the CSV file containing the labels.
    label_column : str
        The name of the label column to extract from the labels file.

    Returns
    -------
    Dataset
        A PyTorch dataset with the features and labels ready for training.
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
        return len(self.features)

    def __getitem__(self, idx):
        """Retrieve the feature vector and label for a given index."""
        return self.features[idx], self.labels[idx]
