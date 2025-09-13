import json
import numpy as np

# Amino acid type lookup
atm_types = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]


def AA2num(S, atm_types=atm_types):
    """Convert a sequence string to a numeric array using the amino acid lookup."""
    return np.array([atm_types.index(i) for i in S])


def back_AA(X, atm_types=atm_types):
    """Convert a numeric array back to a sequence string."""
    X = np.asarray(X)
    return ''.join(atm_types[int(i)] for i in X)


def load_normalization_stats(file_path):
    """Load feature normalization statistics from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def standard_normalize_features(reshaped_features, normalization_stats):
    """Apply standard normalization to the feature vector using precomputed stats."""
    std_normal_dict = normalization_stats['std_normal_dict']
    min_L = normalization_stats['min_L']
    max_L = normalization_stats['max_L']
    maxS = normalization_stats['maxS']

    reshaped_features[:20] /= reshaped_features[20]
    reshaped_features[23] /= reshaped_features[20]
    reshaped_features[24] /= reshaped_features[20]
    reshaped_features[25] /= reshaped_features[20]
    reshaped_features[26] /= reshaped_features[20]
    reshaped_features[28] /= reshaped_features[20]

    reshaped_features[20] = (reshaped_features[20] - min_L) / (max_L - min_L)

    std_norm_indices = {
        'SCD': 21, 'SHD': 22, '|net charge|': 23, 'sum lambda': 24,
        'beads(+)': 25, 'beads(-)': 26, 'shannon_entropy': 27, 'mol wt': 28
    }

    for feat, idx in std_norm_indices.items():
        if feat == 'shannon_entropy':
            reshaped_features[idx] = reshaped_features[idx] / maxS - 1
        else:
            mean_val, std_val = std_normal_dict[feat]
            reshaped_features[idx] = (reshaped_features[idx] - mean_val) / std_val

    return reshaped_features

__all__ = [
    'atm_types', 'AA2num', 'back_AA',
    'load_normalization_stats', 'standard_normalize_features'
]
