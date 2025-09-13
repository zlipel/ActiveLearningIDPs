import json
import os
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler, PowerTransformer
from al_pipeline.features.data_preprocessing import load_dataset
from al_pipeline.models.gpr_model import GPRegressionModel, MultitaskGPRegressionModel

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


def load_gpr_models(model_path, master_path, iteration, model_labels,
                    ehvi_var, explore, seq_id, transform):
    """Load saved multitask GPR models and label scalers."""
    scalers = []
    gpr_file = (
        os.path.join(model_path, f"GPR_iter{iteration}_{ehvi_var}_{explore}_{transform}_TEMP.pt")
        if seq_id > 1 and explore not in ['standard', 'similarity_penalty']
        else os.path.join(model_path, f"GPR_iter{iteration}_{ehvi_var}_{explore}_{transform}.pt")
    )
    features_file = os.path.join(master_path, f"features_gen{iteration}.csv")
    norm_features_file = os.path.join(
        master_path,
        f"features_gen{iteration}_NORM_{ehvi_var}_{explore}_{transform}.csv",
    )
    labels_file = os.path.join(master_path, f"labels_gen{iteration}.csv")
    norm_labels_file = os.path.join(
        master_path,
        f"labels_gen{iteration}_NORM_{ehvi_var}_{explore}_{transform}.csv",
    )

    for label in model_labels:
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found at {features_file}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found at {labels_file}")
        features, labels = load_dataset(features_file, labels_file, label_column=label)
        if label == 'diff' and transform == 'log':
            labels = np.log(labels + 1e-8)
        if transform == 'yeoj':
            scaler = PowerTransformer(method='yeo-johnson')
            scaler.fit_transform(labels.reshape(-1, 1))
        elif transform == 'log':
            scaler = StandardScaler()
            scaler.fit_transform(labels.reshape(-1, 1))
        scalers.append(scaler)

    if not os.path.exists(gpr_file):
        raise FileNotFoundError(f"Checkpoint not found at {gpr_file}")

    if os.path.exists(norm_features_file):
        features_norm = torch.tensor(pd.read_csv(norm_features_file).values).float()
    else:
        raise FileNotFoundError(
            f"Normalized features file not found at {norm_features_file}"
        )

    if os.path.exists(norm_labels_file):
        labels_norm = torch.tensor(pd.read_csv(norm_labels_file).values).float()
    else:
        raise FileNotFoundError(
            f"Normalized labels file not found at {norm_labels_file}"
        )

    if features_norm.size(0) != labels_norm.size(0):
        raise ValueError(
            "Features and labels must have the same number of samples."
        )

    checkpoint = torch.load(gpr_file, weights_only=True)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPRegressionModel(features_norm, labels_norm, likelihood, num_tasks=2)
    model.load_state_dict(checkpoint['model'])

    return model, likelihood, scalers, features_norm


def alpha(sequences, propseqs, seq_id):
    """Similarity penalty based on dot products between sequences."""
    if seq_id == 1:
        return np.ones(sequences.shape[0])
    xidotxk = np.matmul(sequences, propseqs.T)
    magnitude_seq = np.sqrt(np.sum(sequences ** 2, axis=1))
    magnitude_propseq = np.sqrt(np.sum(propseqs ** 2, axis=1))
    full_matrix = 0.5 * (1.0 - xidotxk / np.outer(magnitude_seq, magnitude_propseq))
    full_matrix = np.where(full_matrix == 0, 1e-6, full_matrix) ** (-1)
    return (seq_id - 1) / np.sum(full_matrix, axis=1)


def load_pareto_front(master_path, iteration, columns, scalers,
                       ehvi_variant, exploration, seq_id, transform):
    """Load the Pareto front for a given iteration."""
    labels = (
        pd.read_csv(
            os.path.join(
                master_path,
                f"labels_parent_NORM_{ehvi_variant}_{exploration}_{transform}.csv",
            )
        )
        if exploration not in ['standard', 'similarity_penalty']
        else pd.read_csv(os.path.join(master_path, 'labels_parent.csv'))
    )
    pareto_front = labels[columns].values

    feats = pd.read_csv(
        os.path.join(
            master_path,
            f"features_parent_NORM_{ehvi_variant}_{exploration}_{transform}.csv",
        )
    )

    if exploration in ['kriging_believer', 'constant_liar_min',
                      'constant_liar_mean', 'constant_liar_max']:
        return pareto_front, feats.values

    for i, scaler in enumerate(scalers):
        if i == 0:
            pareto_front[:, i] = scaler.transform(
                pareto_front[:, i].reshape(-1, 1)
            ).flatten()
        elif i == 1:
            pareto_front[:, i] = scaler.transform(
                np.log(pareto_front[:, i] + 1e-8).reshape(-1, 1)
            ).flatten()

    return pareto_front, feats.values


def save_cand_sequence(sequence, fitness, output_folder, cand_id, difference=None):
    """Persist a candidate sequence and its fitness to disk."""
    seq_file = os.path.join(output_folder, f"seq_cand_{cand_id}.txt")
    with open(seq_file, 'w') as f:
        f.write(back_AA(sequence) + '\n')
        f.write(str(fitness) + '\n')
        if difference is not None:
            f.write(str(difference) + '\n')

__all__ = [
    'atm_types', 'AA2num', 'back_AA',
    'load_normalization_stats', 'standard_normalize_features',
    'load_gpr_models', 'alpha', 'load_pareto_front', 'save_cand_sequence'
]
