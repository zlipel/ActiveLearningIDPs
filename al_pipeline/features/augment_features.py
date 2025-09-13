import json
import numpy as np
import torch
import gpytorch
import pandas as pd
import argparse
from time import time
import os
from al_pipeline.models.gpr_model import MultitaskGPRegressionModel
from al_pipeline.selection.geneticalgorithm_m2 import geneticalgorithm_batch as ga
from al_pipeline.features.data_preprocessing import load_dataset
# from utils.ehvi_2d import psi, ehvi_batch
from al_pipeline.selection import ehvi
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from . import sequence_featurizer as sf
import pickle
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.preprocessing import StandardScaler, PowerTransformer

def save_chkpt(model_path, model, optimizer=None, val_losses=None, train_losses=None, trained=False):
    """
    Save a training checkpoint.

    Args:
        model_path (str): The path to save the model to.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        val_losses (list of float): A list containing the validation losses.
        train_losses (list of float): A list containing the training losses.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if trained==True:
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
    else:
        state_dict = {
            'model': model.state_dict()
        }
    torch.save(state_dict, model_path)


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
        if label == 'diff' and 'log' in transform:
            # Convert diffusion coefficient to log scale
            labels = np.log(labels + 1e-8)

        if 'yeoj' in transform:
            scaler = PowerTransformer(method='yeo-johnson')
            scaler.fit_transform(labels.reshape(-1,1))
        elif 'log' in transform:
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
    likelihood.load_state_dict(checkpoint['likelihood']) if 'likelihood' in checkpoint else None
    
    return model, likelihood, scalers 

def standard_normalize_features(reshaped_features, normalization_stats):
    std_normal_dict = normalization_stats['std_normal_dict']
    min_L = normalization_stats['min_L']
    max_L = normalization_stats['max_L']
    maxS = normalization_stats['maxS']

    # Apply standard normalization based on the loaded stats (same as before)
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

def load_normalization_stats(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser(description='Run Genetic Algorithm for intermediate iterations of active learning.')
    parser.add_argument('--gen_folder', type=str, required=True, help='Folder where the genetic algorithm produces children.')
    parser.add_argument('--iter_folder', type=str, required=True, help='Folder where current generation is located, eg features, labels.')
    parser.add_argument('--iteration', type=int, required=True, help='Iteration number for the genetic algorithm.')
    parser.add_argument('--seq_id', type=int, required=True, help='ID of sequence we are generating in the global loop.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the ML models (e.g. GPR and RF models).')
    parser.add_argument('--obj1', type=str, required=True, help='Objective 1 for the Pareto front.')
    parser.add_argument('--obj2', type=str, required=True, help='Objective 2 for the Pareto front.')
    parser.add_argument('--ehvi_variant', type=str, choices=['standard', 'epsilon'], default='epsilon',
                    help='Type of EHVI strategy: standard or epsilon.')
    parser.add_argument('--exploration_strategy', type=str, choices=['similarity_penalty', 'kriging_believer', 'constant_liar_min', 'constant_liar_mean', 'constant_liar_max', 'standard'], default='similarity_penalty',
                    help='Exploration method: similarity_penalty or front_augmentation.')
    parser.add_argument('--transform', type=str, choices=['yeoj', 'log'], default='log',
                    help='Transformation applied to the labels: yeojohnson or log.')
    parser.add_argument("--monte_carlo", type=str, required=False, default=None, help="Whether to use Monte Carlo sampling for EHVI calculation. If provided, it will be used as a flag.")

    args = parser.parse_args()
    db_path = "/home/zl4808/scripts/GENDATA/databases"
    model_name = args.gen_folder.split('/')[6]
    assert model_name in ['HPS_URRY', 'MPIPI', 'HPS_KR', 'MARTINI', 'CALVADOS'], "Model name not recognized. Please check the model name."
    # Initialize the sequence featurizer
    featurizer = sf.SequenceFeaturizer(model_name.lower(), db_path)
    #iteration_folder = args.curr_folder + f'/iteration_{args.iteration}'

    transform = args.transform
    if args.monte_carlo is not None:
        transform = f"{args.transform}_MC"
    feats_total = pd.read_csv(os.path.join(args.iter_folder, f"features_gen{args.iteration}_NORM_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.csv"))
    labels_total = pd.read_csv(os.path.join(args.iter_folder, f"labels_gen{args.iteration}_NORM_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.csv"))

    if args.seq_id == 1:
        seq_file = os.path.join(args.iter_folder, f"seq_gen{args.iteration}.txt")
    else:
        seq_file = os.path.join(args.iter_folder, f"seq_gen{args.iteration}_TEMP_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.txt")

    column_names = feats_total.columns.tolist()

    objectives = [args.obj1, args.obj2]
    print(f"objectives:{objectives}", flush=True)

    normalization_stats = load_normalization_stats(os.path.join(args.iter_folder, f'normalization_stats.json'))
    model, likelihood, scalers = load_gpr_models(args.model_path, args.iter_folder, args.iteration, objectives, args.ehvi_variant, args.exploration_strategy, args.seq_id, transform)
    print("models and stats loaded", flush=True)
    model.eval()
    likelihood.eval()

    children_folder = os.path.join(args.gen_folder, f"children_{args.ehvi_variant}_{args.exploration_strategy}_{transform}")

    child_file = os.path.join(children_folder, f'seq_child_{args.seq_id}.txt')
    #print(f"here are the child files {child_files}", flush=True)
    seqs = []
    with open(child_file, 'r') as f:
        seq = f.readline().strip()
        seqs.append(seq)

    # Featurize and normalize
    features = []
    for seq in seqs:
        #print(seq, flush=True)
        raw_feats = featurizer.featurize(seq)
        norm_feats = standard_normalize_features(np.asarray(raw_feats), normalization_stats)
        features.append(norm_feats)

    features = np.array(features)

    new_frame = pd.DataFrame(features, columns=column_names)


  
    # concatenate with the total features
    feats_new = pd.concat([feats_total, new_frame], ignore_index=True)
    # Save the updated features to the generation folder as temp 
    feats_new.to_csv(os.path.join(args.iter_folder, f'features_gen{args.iteration}_NORM_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.csv'), index=False)

    # Predict objevtives using the GPR model
    print(f"Predicting objectives for sequence {args.seq_id} in iteration {args.iteration}.", flush=True)

    if args.exploration_strategy == 'kriging_believer':
        # Use the GPR model to predict the objectives
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = likelihood(model(torch.tensor(features).float())).mean.detach().numpy()
    elif args.exploration_strategy in ['constant_liar_min', 'constant_liar_mean', 'constant_liar_max']:
        # first, get the generation we are at
        gen = args.iteration
        num_rows = 120 + args.iteration * 48  # 120 is the number of rows in the first generation, 48 is the number of rows added in each subsequent generation
        
        # choose either min, mean or max of the labels before our sequential batch generation as a constant lie

        if args.exploration_strategy == 'constant_liar_min':
            preds = labels_total.iloc[:num_rows].min().values
        elif args.exploration_strategy == 'constant_liar_mean':
            preds = labels_total.iloc[:num_rows].mean().values
        elif args.exploration_strategy == 'constant_liar_max':
            preds = labels_total.iloc[:num_rows].max().values
        preds = np.tile(preds, (features.shape[0], 1))


    print(f"Predictions : {preds[0,0], preds[0,1], type(preds[0,0]), type(preds[0,1])}", flush=True)

    # append the new predictions to the normalized labels w/ same columns as labels_total

    new_labels = pd.DataFrame(preds, columns=labels_total.columns)
    # Concatenate with the total labels
    labels_total = pd.concat([labels_total, new_labels], ignore_index=True)
    # Save the updated labels to the generation folder as temp
    labels_total.to_csv(os.path.join(args.iter_folder, f'labels_gen{args.iteration}_NORM_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.csv'), index=False)

    # now that we have the new files, we can retrain our GPR model on the augmented dataset

    print(f"Retraining GPR model with new features and labels for iteration {args.iteration}.", flush=True)

    train_X = torch.tensor(feats_new.values).float()
    train_y = torch.tensor(labels_total.values).float()
    # Create a new GPR model and likelihood
    likelihood_new = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model_new = MultitaskGPRegressionModel(train_X, train_y, likelihood, num_tasks=2)

    # load hyperparameters from the previous model
    model_new.load_state_dict(model.state_dict())

    # Set the model and likelihood to training mode
    model_new.train()
    likelihood_new.train()
    # Set the model to use the same device as the data
    device = train_X.device if train_X.is_cuda else 'cpu'
    model_new.to(device)
    likelihood_new.to(device)


    # Define the optimizer
    optimizer = torch.optim.Adam(model_new.parameters(), lr=0.1)  # Use a lower learning rate for fine-tuning
    # Define the loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_new, model_new)
    # Training loop
    num_epochs = 500  # Adjust the number of epochs as needed
    for epoch in range(num_epochs):

        optimizer.zero_grad()  # Zero gradients from previous iteration
        output = model_new(train_X)  # Forward pass
        loss = -mll(output, train_y)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the model parameters
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}", flush=True)
    # Save the updated model
    gpr_file = os.path.join(args.model_path, f'GPR_iter{args.iteration}_{args.ehvi_variant}_{args.exploration_strategy}_{transform}_TEMP.pt') 

    # now save model and likelihood

    likelihood_new.eval()
    model_new.eval()

    save_chkpt(gpr_file, model_new)
    print(f"Updated GPR model saved to {gpr_file}", flush=True)


    print(f"Saving sequence {seqs[0]} to {seq_file}", flush=True)
    # Save the sequence to the generation folder as temp
    with open(seq_file, 'r') as f:
        existing_seqs = f.readlines()
    existing_seqs = [line.strip() for line in existing_seqs]
    
    existing_seqs.append(seqs[0])  # Append the new sequence

    #if seq_file.endswith('_TEMP.txt'):
    if 'TEMP' in seq_file:
        with open(seq_file, 'w') as f:
            for seq in existing_seqs:
                f.write(seq + '\n')
    else:
        # If the file is not a temp file, we create a new one
        with open(os.path.join(args.iter_folder, f'seq_gen{args.iteration}_TEMP_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.txt'), 'w') as f:
            for seq in existing_seqs:
                f.write(seq + '\n')

if __name__ == "__main__":
    main()
     