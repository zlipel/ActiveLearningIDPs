
import numpy as np
import torch
import gpytorch
import argparse
from time import time
import os
from .geneticalgorithm_m2 import geneticalgorithm_batch as ga
from . import ehvi
from joblib import Parallel, delayed
from al_pipeline.features import sequence_featurizer as sf
from .utils import (
    AA2num,
    back_AA,
    load_normalization_stats,
    standard_normalize_features,
    load_gpr_models,
    alpha,
    load_pareto_front,
    save_cand_sequence,
)






"""Sequential GA candidate generation and EHVI evaluation."""

import json
import numpy as np
import torch
import gpytorch
import pandas as pd
import argparse
from time import time
import os
from al_pipeline.models.gpr_model import GPRegressionModel, MultitaskGPRegressionModel
from .geneticalgorithm_m2 import geneticalgorithm_batch as ga
from al_pipeline.features.data_preprocessing import load_dataset
from . import ehvi
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from al_pipeline.features import sequence_featurizer as sf
import pickle
from sklearn.preprocessing import StandardScaler, PowerTransformer



atm_types = ['A',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'V',
 'W',
 'Y']

# convert string to a numpy array
def AA2num(S, atm_types=atm_types):
    """Encode a sequence string into numeric indices."""
    X = [atm_types.index(i) for i in S]
    return np.array(X)


def back_AA(X, atm_types=atm_types):
    """Decode numeric indices back to an amino acid string."""
    X = np.asarray(X)
    AA_str = [atm_types[int(i)] for i in X]
    return ''.join(AA_str)


def load_normalization_stats(file_path):
    """Load feature normalization statistics from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)

def standard_normalize_features(reshaped_features, normalization_stats):
    """Normalize a raw feature vector using precomputed statistics."""
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


def load_gpr_models(model_path, master_path, iteration, model_labels, ehvi_var, explore, seq_id, transform):
    """
    Load the GPR models from the specified path.
    
    Args:
        model_path: Path to the saved GPR models.
        iteration: The iteration number for the active learning process.
        model_labels: List of labels for the GPR models.
        
    Returns: 
        List of GPR models, likelihoods, and scalers (for labels).
    """

    # Check if model file exists

    scalers = []
    labels_total = []


    gpr_file = os.path.join(model_path,f'GPR_iter{iteration}_{ehvi_var}_{explore}_{transform}_TEMP.pt') if seq_id > 1 and explore not in ['standard', 'similarity_penalty'] \
        else os.path.join(model_path,f'GPR_iter{iteration}_{ehvi_var}_{explore}_{transform}.pt')
    
    # features_file = os.path.join(master_path,f'features_gen{iteration}_TEMP.csv') if seq_id > 1 and explore == 'front_augmentation' \
    #     else os.path.join(master_path,f'features_gen{iteration}.csv')

    features_file = os.path.join(master_path,f'features_gen{iteration}.csv')
    norm_features_file = os.path.join(master_path,f'features_gen{iteration}_NORM_{ehvi_var}_{explore}_{transform}.csv')
    
    labels_file = os.path.join(master_path,f'labels_gen{iteration}.csv') 
    norm_labels_file = os.path.join(master_path,f'labels_gen{iteration}_NORM_{ehvi_var}_{explore}_{transform}.csv')


    # next block is to load the 

    for label in model_labels:
        
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found at {features_file}")
        
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found at {labels_file}")
        
        # load the training data    
        features, labels = load_dataset(features_file, \
                                                labels_file, \
                                                    label_column=label)
        if label == 'diff' and transform == 'log':
            # Convert diffusion coefficient to log scale
            labels = np.log(labels + 1e-8)

        if transform == 'yeoj':
            scaler = PowerTransformer(method='yeo-johnson')
            scaler.fit_transform(labels.reshape(-1,1))
        elif transform == 'log':
            scaler = StandardScaler()
        # Fit the scaler to the labels
            scaler.fit_transform(labels.reshape(-1, 1))
        
        scalers.append(scaler)
        
    
    if not os.path.exists(gpr_file):
            raise FileNotFoundError(f"Checkpoint not found at {gpr_file}")
    
    # now load the normalized features and labels up to that point 
    if os.path.exists(norm_features_file):
        features_norm = torch.tensor(pd.read_csv(norm_features_file).values).float()
    else:
        raise FileNotFoundError(f"Normalized features file not found at {norm_features_file}")
    
    if os.path.exists(norm_labels_file):
        labels_norm = torch.tensor(pd.read_csv(norm_labels_file).values).float()
    else:   
        raise FileNotFoundError(f"Normalized labels file not found at {norm_labels_file}")
    # Check if features and labels have the same number of samples
    if features_norm.size(0) != labels_norm.size(0):
        raise ValueError(f"Features and labels must have the same number of samples. Found {features_norm.size(0)} features and {labels_norm.size(0)} labels.")


    # Load the saved state dictionary from file
    checkpoint = torch.load(gpr_file, weights_only=True)

        # Instantiate the likelihood and model classes
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = MultitaskGPRegressionModel(features_norm, labels_norm, likelihood, num_tasks=2) 

        # Load the model and optimizer state dicts
    model.load_state_dict(checkpoint['model'])
    
    return model, likelihood, scalers, features_norm


def alpha(sequences, propseqs, seq_id):
    """
    Calculate the similarity penalty for a list of sequences based on the dot product of the sequences
    and previously proposed sequences.

    Args:
        sequences: numpy array of sequences to calculate similarity penalty for
        propseqs: numpy array of previously proposed sequences
        seq_id: ID of the current sequence being generated through the 96 GA runs

    Returns: numpy array of similarity penalty values for each sequence
    """
    if seq_id == 1:
        return np.ones(sequences.shape[0])
    else:
        xidotxk = np.matmul(sequences, propseqs.T) # matrix with entries x_i dot x_k
        magnitude_seq = np.sqrt(np.sum(sequences**2, axis=1)) # magnitude of candidates
        magnitude_propseq = np.sqrt(np.sum(propseqs**2, axis=1)) # magnitude of previously proposed

        full_matrix = 0.5 * (1.0 - xidotxk / np.outer(magnitude_seq, magnitude_propseq))
        
        # Replace any zero entries with a small positive value to avoid division by zero
        full_matrix = np.where(full_matrix == 0, 1e-6, full_matrix) ** (-1)

        return (seq_id - 1)/np.sum(full_matrix, axis=1) # similarity penalty per candidate,

        


def fitness_function_batch(sequences, seq_id, featurizer, augmented_front, propseq_arr, gpr_model, likelihood, normalization_stats, front, exploration="similarity_penalty"):
    """
    Batch fitness function using the GPR model to predict B2 values for a list of sequences.

    Args:
        sequences: List of sequences to predict fitness values for.
        seq_id: ID of the current sequence being generated through the 96 GA runs
        pareto_front: Pareto front of B2 and diff values
        propseqs: List of previously proposed sequences (eg up to seq_id - 1)
        gpr_models: List of GPR models for B2 and diff
        likelihoods: List of likelihoods for B2 and diff
        normalization_stats: Normalization statistics for the features
        front: Upper or lower Pareto front

    Returns: List of fitness values for each sequence 
    """
    # Featurize all sequences and normalize them
    seq_list = [back_AA(seq) for seq in sequences]
    #seq_feats_list = [featurizer.featurize(seq) for seq in sequences]
    seq_feats_list = Parallel(n_jobs=-1)(delayed(featurizer.featurize)(seq) for seq in seq_list)
    seq_feats_list_arr = [np.asarray(feats) for feats in seq_feats_list]
    seq_feats_normal_list = [standard_normalize_features(feats, normalization_stats) for feats in seq_feats_list_arr]
    seq_arr = np.array(seq_feats_normal_list)

    # Convert all features to a tensor batch
    seq_tensor_batch = torch.tensor(seq_arr).float()

    # Perform batch prediction using the GPR model
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     pred_B2_batch, std_B2_batch     = likelihood(gpr_model(seq_tensor_batch)).mean[:,0], likelihood(gpr_model(seq_tensor_batch)).stddev[:,0]
    #     pred_diff_batch, std_diff_batch = likelihood(gpr_model(seq_tensor_batch)).mean[:,1], likelihood(gpr_model(seq_tensor_batch)).stddev[:,1]


    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = gpr_model(seq_tensor_batch)            # latent posterior over f
        mu   = post.mean                              # shape [B, 2]
        std  = post.stddev                            # shape [B, 2]  (sqrt of marginal variances)

    pred_B2_batch   = mu[:, 0]
    std_B2_batch    = std[:, 0]
    pred_diff_batch = mu[:, 1]
    std_diff_batch  = std[:, 1]

    # reshape the predictions and standard deviations

    pred_B2_reshaped   = pred_B2_batch.numpy().flatten() #.reshape(-1, 1)
    pred_diff_reshaped = pred_diff_batch.numpy().flatten() #.reshape(-1, 1)

    std_B2 = std_B2_batch.numpy().flatten() 
    std_diff = std_diff_batch.numpy().flatten()

    ##### Calculate EHVI ####

    # ehvi calc
    # rather than feeding in the -B2 vs D, we feed B2 vs -D in order to maximize the EHVI by doing the 
    # equivalent minimization problem of B2 vs -D (maximizing -B2 vs D)
    if front == 'lower':
        ehvi_values = ehvi.ehvi_maximization(pred_B2_reshaped, std_B2, pred_diff_reshaped, std_diff, augmented_front)
    elif front == 'upper':
        ehvi_values = ehvi.ehvi_maximization(-pred_B2_reshaped, std_B2, -pred_diff_reshaped, std_diff, augmented_front)
    else:
        raise ValueError("Invalid front type. Choose 'upper' or 'lower'.")
 
    #print(f"EHVI values for seq_id {seq_id}: {ehvi_values}", flush=True)
    ##### Calculate similarity penalty using alpha ######
    if seq_id > 1 and exploration == 'similarity_penalty':
        alpha_values = alpha(seq_arr, propseq_arr, seq_id)  # Apply similarity penalty
        #print(alpha_values, flush=True)
    else:  
        alpha_values = np.ones(len(pred_B2_reshaped))
    #print(ehvi_values, flush=True)
    # Return the inverse-transformed outputs as a list
    return  -ehvi_values*alpha_values # fitness function (minimize negative EHVI)



def load_pareto_front(master_path, iteration, columns, scalers, ehvi_variant, exploration, seq_id, transform):
    """
    Load the Pareto front of B2 and diff values for the specified iteration.
    Args:
        master_path: Path to the master folder for the project.
        iteration: The iteration number for the active learning process.
        columns: List of columns to load from the labels file.
        scalers: List of scalers to standardize the labels.
        
    Returns: 
        Numpy array of the Pareto front values in the shape of [N, 2] where N is the number of sequences.
    """

    labels = pd.read_csv(os.path.join(master_path,f'labels_parent_NORM_{ehvi_variant}_{exploration}_{transform}.csv')) if exploration not in  ['standard', 'similarity_penalty'] \
        else pd.read_csv(os.path.join(master_path,f'labels_parent.csv'))
    pareto_front = labels[columns].values

    feats = pd.read_csv(os.path.join(master_path,f'features_parent_NORM_{ehvi_variant}_{exploration}_{transform}.csv'))
    
    if exploration == 'kriging_believer' or exploration == 'constant_liar_min' or exploration == 'constant_liar_mean' or exploration == 'constant_liar_max':
            # If we are using the front augmentation, we need to standardize the Pareto front based on the generation 
        return pareto_front, feats.values
    else:
        for i, scaler in enumerate(scalers):
            if i == 0:
                pareto_front[:,i] = scaler.transform(pareto_front[:,i].reshape(-1,1)).flatten() # convert to standard scaling based on generation
            elif i == 1:
                pareto_front[:,i] = scaler.transform(np.log(pareto_front[:,i]+1e-8).reshape(-1,1)).flatten()
        
        return pareto_front, feats.values


def save_cand_sequence(sequence, fitness, output_folder, cand_id):
    """
    Save the candidate sequence as a text file with the specified candidate ID.
    
    Args:
        sequence: The generated candidate sequence.
        output_folder: Folder where the sequence should be saved.
        cand_id: Unique ID for each candidate (used in the filename).
    """
    seq_file = os.path.join(output_folder, f"seq_cand_{cand_id}.txt")

    # Save sequence to the file
    with open(seq_file, 'w') as f:
        f.write(back_AA(sequence) + '\n')
        f.write(str(fitness) + '\n')



def main():
    timeinit = time()
    parser = argparse.ArgumentParser(description='Run Genetic Algorithm for intermediate iterations of active learning.')
    parser.add_argument('--gen_folder', type=str, required=True, help='Folder where the genetic algorithm produces children.')
    parser.add_argument('--iter_folder', type=str, required=True, help='Folder where current generation is located, eg features, labels.')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number for the genetic algorithm.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the ML models (e.g. GPR and RF models).')
    parser.add_argument('--obj1', type=str, required=True, help='Objective 1 for the Pareto front.')
    parser.add_argument('--obj2', type=str, required=True, help='Objective 2 for the Pareto front.')
    parser.add_argument('--cand_id', type=int, required=True, help='ID of sequence to output.')
    parser.add_argument('--seq_id', type=int, required=True, help='ID of sequence we are generating in the global loop.')
    parser.add_argument('--verbose', type=str, required=True, help='Option to output progress bar (True/False).')
    parser.add_argument('--front', type=str, required=True, help='Upper or lower Pareto front.')
    parser.add_argument('--ehvi_variant', type=str, choices=['standard', 'epsilon'], default='epsilon',
                    help='Type of EHVI strategy: standard or epsilon.')
    parser.add_argument('--exploration_strategy', type=str, choices=['similarity_penalty', 'kriging_believer', 'constant_liar_min', 'constant_liar_mean', 'constant_liar_max', 'standard'], default='similarity_penalty',
                    help='Exploration method: similarity_penalty or front_augmentation.')
    parser.add_argument('--transform', type=str, choices=['yeoj', 'log'], default='log', help='Transformation applied to the labels.')

    args = parser.parse_args()

    db_path = "/home/zl4808/scripts/GENDATA/databases"
    model_name = args.gen_folder.split('/')[6]
    assert model_name in ['HPS_URRY', 'MPIPI', 'HPS_KR', 'MARTINI', 'CALVADOS'], "Model name not recognized. Please check the model name."

    # Initialize the sequence featurizer
    featurizer = sf.SequenceFeaturizer(model_name.lower(), db_path)
    #iteration_folder = args.curr_folder + f'/iteration_{args.iteration}'

    objectives = [args.obj1, args.obj2]

    # Load the GPR models and normalization stats
    model, likelihood, scalers, feats_tensor = load_gpr_models(args.model_path, args.iter_folder, args.iteration, objectives, args.ehvi_variant, args.exploration_strategy, args.seq_id, args.transform)
    normalization_stats = load_normalization_stats(os.path.join(args.iter_folder, f'normalization_stats.json'))

    model.eval()
    likelihood.eval()

    #with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #    vars = likelihood(model(torch.tensor(feats_tensor).float())).variance.sqrt().numpy()
    
    #val1 = np.mean(vars[:,0]) if args.front == 'upper' else -np.mean(vars[:,0])
    #val2 = np.mean(vars[:,1]) if args.front == 'upper' else -np.mean(vars[:,1])
    #print(f"Epsilons for FULL DAT for seq {args.seq_id}, candidate {args.cand_id}: {(val1, val2)}", flush=True)


    # # Load the RF model
    # rf_model = joblib.load(args.model_path + f'/RF_psp_iter{args.iteration}.pkl')

    # Load parent sequences
    seq_file = os.path.join(args.gen_folder, f'sequences_parent_TEMP_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}.txt') if args.seq_id > 1 and args.exploration_strategy not in ['standard', 'similarity_penalty'] \
        else os.path.join(args.gen_folder, 'sequences_parent.txt')
    with open(seq_file, 'r') as f:
        parent_seqs = [line.strip() for line in f]
    init_pop = [list(AA2num(seq)) for seq in parent_seqs]

    # to use if we are using the similarity penalty
    previous_candidates = []
        
    if args.seq_id > 1:
        for i in range(args.seq_id-1):
            with open(os.path.join(args.gen_folder, f"children_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}", f"seq_child_{i+1}.txt"), 'r') as f:
                previous_candidates.append(f.readline().strip())

        previous_candidates = [list(AA2num(seq)) for seq in previous_candidates]  


    if len(previous_candidates)>0:
        prop_seq_list = [back_AA(seq) for seq in previous_candidates]  
        propseq_feats_list = [featurizer.featurize(seq) for seq in prop_seq_list]
        propseq_feats_list_arr = [np.asarray(seq) for seq in propseq_feats_list]
        propseq_feats_normal_list = [standard_normalize_features(feats, normalization_stats) for feats in propseq_feats_list_arr]
        propseq_arr = np.array(propseq_feats_normal_list)
    
        if propseq_arr.ndim == 1:
                propseq_arr = propseq_arr.reshape(1, -1) 
    else:
        propseq_arr = previous_candidates

    pareto_front, pareto_feats = load_pareto_front(args.gen_folder, args.iteration, objectives, scalers, args.ehvi_variant, args.exploration_strategy, args.seq_id, args.transform)

    pareto_input = np.zeros_like(pareto_front)

    if args.ehvi_variant == 'epsilon':

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = model(torch.tensor(pareto_feats).float())  # NOTE: model(...), not likelihood(model(...))
            std = post.stddev.cpu().numpy()                  # shape [N, 2], noise-free marginal stds

        # mean (or median) uncertainty per objective along the front
        sigma_bar = np.median(std, axis=0)                     # shape (2,)  # use np.median for more robustness

        # direction: push towards the non-dominated region
        sign = +2.0 if args.front == 'upper' else -2.0

        # scale: keep modest to avoid overshooting (which can yield EHVI ~ 0)
        k = 1.0  # try 0.5–1.0 for MC-EHVI stability

        # raw epsilon vector
        eps = sign * k * sigma_bar                           # shape (2,)

        # cap by a fraction of the front's span to prevent "unbeatable" pushes

        print(f"Raw epsilons for PARETO FRONT for seq {args.seq_id}, candidate {args.cand_id}: {(eps)}", flush=True)
    

        # shift the entire front once by this epsilon
        pareto_input = pareto_front.copy()
        pareto_input[:, 0] += eps[0]
        pareto_input[:, 1] += eps[1]

        print(f"Capped Epsilons for PARETO FRONT for seq {args.seq_id}, candidate {args.cand_id}: {(eps)}", flush=True)
    else:
        pareto_input = pareto_front.copy()

    
    augmented_front = ehvi.front_augmentation(pareto_input, args.front, epsilons=None if args.ehvi_variant == 'standard' else (eps[0], eps[1]))

   

    ga_B2 = ga(function=lambda seq: fitness_function_batch(seq, args.seq_id, featurizer, augmented_front, \
                                                           propseq_arr, model, likelihood, \
                                                            normalization_stats, args.front, exploration=args.exploration_strategy),
               algorithm_parameters={
                   'max_num_iteration': 200,
                   'population_size': len(init_pop),
                   'mutation_probability': 0.5,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'deletion_probability': 0.2,
                   'growth_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type': 'uniform',
                   'max_iteration_without_improv': None
               }, 
               convergence_curve=False, 
               progress_bar=False)

    # Run the GA to generate new sequences
    ga_B2.run(init_pop=init_pop)

    # Get the best sequence and its fitness for this run
    output_sequence = ga_B2.best_variable
    output_fitness = ga_B2.best_function


    candidates_folder = os.path.join(args.gen_folder, f"candidates_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}")
    # Save the best sequence and its fitness
    save_cand_sequence(output_sequence, output_fitness, candidates_folder, args.cand_id)

    
if __name__ == "__main__":
    main()
