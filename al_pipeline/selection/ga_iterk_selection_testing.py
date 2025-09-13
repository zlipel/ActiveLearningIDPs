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
        span = np.ptp(pareto_front, axis=0).clip(min=1e-12)  # per-objective range
        cap  = 0.2 * span                                    # 20% cap; tune 0.1–0.3
        print(f"Span for PARETO FRONT for seq {args.seq_id}, candidate {args.cand_id}: {(span)}", flush=True)
        print(f"Raw epsilons for PARETO FRONT for seq {args.seq_id}, candidate {args.cand_id}: {(eps)}", flush=True)
        eps = np.clip(eps, -cap, cap)

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
