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
from pygmo import hypervolume
from .utils import (
    AA2num,
    back_AA,
    load_normalization_stats,
    standard_normalize_features,
    load_gpr_models,
    load_pareto_front,
    save_cand_sequence,
)








def minmax_scale(X, ref_min=None, ref_max=None):
    if ref_min is None:
        ref_min = np.min(X, axis=0)
    if ref_max is None:
        ref_max = np.max(X, axis=0)
    return (X - ref_min) / (ref_max - ref_min), ref_min, ref_max

def dominates(sol1, sol2, kind = ['max', 'max']):
    """
    Check if sol1 dominates sol2 for maximizing D and minimizing B2.
    """
    obj1 = kind[0]
    obj2 = kind[1]

    if obj1 == 'min':
        if obj2 == 'max':
            return (sol1[0] <= sol2[0] and sol1[1] >= sol2[1]) and (sol1[0] < sol2[0] or sol1[1] > sol2[1])
        elif obj2 == 'min':
            return (sol1[0] <= sol2[0] and sol1[1] <= sol2[1]) and (sol1[0] < sol2[0] or sol1[1] < sol2[1])
    elif obj1 == 'max':
        if obj2 == 'min':
            return (sol1[0] >= sol2[0] and sol1[1] <= sol2[1]) and (sol1[0] > sol2[0] or sol1[1] < sol2[1])
        elif obj2 == 'max':
            return (sol1[0] >= sol2[0] and sol1[1] >= sol2[1]) and (sol1[0] > sol2[0] or sol1[1] > sol2[1])
    else:   
        raise ValueError("Invalid objective types.")
    
def filter_nondominated(points):
    """
    Returns a boolean mask of nondominated points (for minimization).
    """
    num_points = points.shape[0]
    is_efficient = np.ones(num_points, dtype=bool)
    
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue
            # Check if point j dominates point i
            if dominates(points[j], points[i], kind=['min', 'min']):
                is_efficient[i] = False
                break  # No need to check others
    return is_efficient

def refpoint(pareto_front, front = 'upper'):
    """
    Calculate the reference point for the hypervolume calculation based on the Pareto front.
    
    Args:
        pareto_front: Numpy array of the Pareto front values in the shape of [N, 2] where N is the number of sequences.
        front: 'upper' or 'lower' Pareto front.
        
    Returns:
        Reference point as a tuple (rB, rD).
    """
    minB = np.min(pareto_front[:,0])
    minD = np.min(pareto_front[:,1])

    maxB = np.max(pareto_front[:,0])
    maxD = np.max(pareto_front[:,1])

    if front =='upper':
        rB = minB - 0.5*np.abs(minB)
        rD = minD - 0.5*np.abs(minD)
        return np.array([-1*rB, -1*rD])
    elif front == 'lower':
        rB = maxB + 0.5*np.abs(maxB)
        rD = maxD + 0.5*np.abs(maxD)
        return np.array([rB, rD])
    else:
        raise ValueError("Invalid front type. Choose 'upper' or 'lower'.") 

def make_ref_point_min_space(pmin, margin=0.15):  # try 0.15–0.30 early on
    worst = pmin.max(axis=0)                    # component-wise worst (min-space)
    span  = pmin.max(axis=0) - pmin.min(axis=0)
    r = worst + margin * np.maximum(span, 1e-9)
    return r


# def monte_carlo_ehvi_batch(candidate_tensor, model, pareto_front, ref_point, base_hv, n_samples=128, front='upper'):
#     """
#     Compute EHVI via Monte Carlo for a batch of candidates using a multi-task GPR.
#     """
#     B = candidate_tensor.size(dim=0)

#     ehvi_vals = np.zeros(B)

#     threshold = 50
#     max_iter = 20

#     iter = 0
#     enough = True
#     # while enough == True or iter < max_iter:

#     posterior = model(candidate_tensor)
#     samples = posterior.rsample(torch.Size([n_samples]))  # shape (n_samples, B, 2)

#     for i in range(B):
#         improvements = []
#         for j in range(n_samples):
#             sample_point = samples[j, i, :].detach().cpu().numpy()

#             if front == 'upper':
#                 sample_point *= -1

#             # Quick dominance check
#             if any(dominates(p, sample_point, kind=['min', 'min']) for p in pareto_front):
#                 improvements.append(0.0)
#                 continue

#             # if sample is outside the bounded region, skip it and don't count it to hvolume:
#             if np.any(sample_point > ref_point):
#                 continue
            

#             extended_front = np.vstack([pareto_front, sample_point])
#             nd_mask = filter_nondominated(extended_front)
#             nd_front = extended_front[nd_mask]
#             hv = hypervolume(nd_front).compute(ref_point)
#             improvements.append(hv - base_hv)

    
#         if len(improvements) < threshold:
#             #enough = False
#             ehvi_vals[i] = 0.0
#         else:
#             ehvi_vals[i] = np.mean(improvements)
            

#     return ehvi_vals


def monte_carlo_ehvi_batch_adaptive(candidate_tensor, model, pareto_front, ref_point, base_hv, 
                            min_samples=64, 
                            max_samples=512,
                            chunk_size=128,
                            stderr_tol=1e-4,
                            front='upper'):
    """
    Compute EHVI via Monte Carlo for a batch of candidates using a multi-task GPR.
    """
    B = candidate_tensor.size(dim=0)
    improvements = [[] for _ in range(B)]  # List of lists to store improvements for each candidate

    ehvi_vals = np.zeros(B)

    threshold = 50
    max_iter = 20

    iter = 0
    enough = True
    # while enough == True or iter < max_iter:

    drawn = np.zeros(B, dtype=int)
    done = np.zeros(B, dtype=bool)
    total = 0

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(candidate_tensor)

    while not np.all(done) and total < max_samples:

        S = min(chunk_size, max_samples - total)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            samples = posterior.rsample(torch.Size([S]))  # shape (n_samples, B, 2)
        s_np = samples.detach().cpu().numpy()

        if front == 'upper':
            s_np *= -1

        for i in range(B):
            
            if done[i]:
                continue

            for s in s_np[:, i, :]:
    

                if np.any(s >= ref_point) or any(dominates(p, s, kind=['min', 'min']) for p in pareto_front):
                    improvements[i].append(0.0)
                    continue
                

                extended_front = np.vstack([pareto_front, s])
                nd_mask = filter_nondominated(extended_front)
                nd_front = extended_front[nd_mask]
                hv = hypervolume(nd_front).compute(ref_point)
                improvements[i].append(hv - base_hv)

            drawn[i] +=  S
            if drawn[i] >= min_samples:
                arr = np.asarray(improvements[i])
                stderr = arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.inf
                if stderr < stderr_tol:
                    done[i] = True


        total += S

    for i in range(B):
        ehvi_vals[i] = np.mean(improvements[i]) if len(improvements[i]) > 0 else 0.0
                
    return ehvi_vals

def monte_carlo_ehvi_batch(candidate_tensor, model, pareto_front, ref_point, base_hv,
                           front='upper', min_samples=64, max_samples=548, stderr_tol=1e-3):
    """
    Compute EHVI via adaptive Monte Carlo for a batch of candidates using a multi-task GPR.

    Args:
        candidate_tensor (Tensor): (B, D) tensor of input features
        model: GPyTorch multi-task GP model
        pareto_front (ndarray): Current Pareto front, shape (N, 2)
        ref_point (ndarray): Reference point for HV computation
        base_hv (float): HV of current Pareto front
        front (str): 'upper' or 'lower'
        min_samples (int): Minimum MC samples
        max_samples (int): Max MC samples per candidate
        stderr_tol (float): Tolerance for standard error of EHVI estimate

    Returns:
        ehvi_vals (ndarray): EHVI estimates for each candidate, shape (B,)
    """
    B = candidate_tensor.size(0)
    ehvi_vals = np.zeros(B)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model(candidate_tensor)

    for i in range(B):
        improvements = []
        sample_count = 0
        converged = False

        while not converged and sample_count < max_samples:
            sample = posterior.rsample(torch.Size([1]))[0, i, :].detach().cpu().numpy()

            if front == 'upper':
                sample *= -1

            if any(dominates(p, sample, kind=['min', 'min']) for p in pareto_front):
                improvements.append(0.0)
            elif np.any(sample >= ref_point):
                continue
            else:
                extended_front = np.vstack([pareto_front, sample])
                nd_mask = filter_nondominated(extended_front)
                nd_front = extended_front[nd_mask]
                hv = hypervolume(nd_front).compute(ref_point)
                improvements.append(hv - base_hv)

            sample_count += 1

            if sample_count >= min_samples:
                stderr = np.std(improvements) / np.sqrt(len(improvements))
                if stderr < stderr_tol:
                    converged = True

        if len(improvements) < min_samples:
            ehvi_vals[i] = 0.0
        else:
            ehvi_vals[i] = np.mean(improvements)

    return ehvi_vals



def fitness_function_batch(sequences, featurizer, pareto_input, hv_base, ref_point, gpr_model, normalization_stats, front):
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


    ##### Calculate EHVI ####
    ehvi_values = monte_carlo_ehvi_batch_adaptive(seq_tensor_batch, gpr_model, pareto_input, ref_point, hv_base,
                            front=front)

    if np.any(ehvi_values > 0.75):
        print(f"EHVI values for sequences: {ehvi_values}", flush=True)

    return  -1*ehvi_values # fitness function (minimize negative EHVI)






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

    transform = args.transform + '_MC' 

    # Load the GPR models and normalization stats
    model, likelihood, scalers, _ = load_gpr_models(
        args.model_path,
        args.iter_folder,
        args.iteration,
        objectives,
        args.ehvi_variant,
        args.exploration_strategy,
        args.seq_id,
        transform,
    )
    normalization_stats = load_normalization_stats(
        os.path.join(args.iter_folder, f'normalization_stats.json')
    )

    model.eval()
    likelihood.eval()


    # Load parent sequences
    seq_file = os.path.join(args.gen_folder, f'sequences_parent_TEMP_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.txt') if args.seq_id > 1 and args.exploration_strategy not in ['standard', 'similarity_penalty'] \
        else os.path.join(args.gen_folder, 'sequences_parent.txt')
    with open(seq_file, 'r') as f:
        parent_seqs = [line.strip() for line in f]
    init_pop = [list(AA2num(seq)) for seq in parent_seqs]

    # to use if we are using the similarity penalty
    previous_candidates = []
        
    if args.seq_id > 1:
        for i in range(args.seq_id-1):
            with open(os.path.join(args.gen_folder, f"children_{args.ehvi_variant}_{args.exploration_strategy}_{transform}", f"seq_child_{i+1}.txt"), 'r') as f:
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

    pareto_front, pareto_feats = load_pareto_front(args.gen_folder, args.iteration, objectives, scalers, args.ehvi_variant, args.exploration_strategy, args.seq_id, transform)

    pareto_input = np.zeros_like(pareto_front)

    if args.ehvi_variant == 'epsilon':


        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = model(torch.tensor(pareto_feats).float())  # NOTE: model(...), not likelihood(model(...))
            std = post.stddev.cpu().numpy()                  # shape [N, 2], noise-free marginal stds
            mmvn = post.to_data_independent_dist()
            C = mmvn.lazy_covariance_matrix.to_dense().cpu().numpy()  # [N, 2, 2]
        rho = C[:,0,1] / np.sqrt(C[:,0,0]*C[:,1,1])
        print('avg rho:', np.mean(rho), 'median rho:', np.median(rho), flush=True)

        # mean (or median) uncertainty per objective along the front
        sigma_bar = np.median(std, axis=0)                     # shape (2,)  # use np.median for more robustness

        # direction: push towards the non-dominated region
        sign = +1.0 if args.front == 'upper' else -1.0

        # scale: keep modest to avoid overshooting (which can yield EHVI ~ 0)
        k = 0.75  # try 0.5–1.0 for MC-EHVI stability

        # raw epsilon vector
        eps = sign * k * sigma_bar                           # shape (2,)

        # cap by a fraction of the front's span to prevent "unbeatable" pushes
        span = np.ptp(pareto_front, axis=0).clip(min=1e-12)  # per-objective range
        cap  = 0.2 * span                                    # 20% cap; tune 0.1–0.3
        eps = np.clip(eps, -cap, cap)

        # shift the entire front once by this epsilon
        pareto_input = pareto_front.copy()
        pareto_input[:, 0] += eps[0]
        pareto_input[:, 1] += eps[1]

        print(f"Epsilons for PARETO FRONT for seq {args.seq_id}, candidate {args.cand_id}: {(eps)}", flush=True)

    else:
        pareto_input = pareto_front.copy()

    # get ref point for the hypervolume calculation
    ref_point = refpoint(pareto_input, front=args.front)


    if args.front == 'upper':
        pareto_input *= -1  # transform to third quadrant for minimization

    #ref_point = make_ref_point_min_space(pareto_input, margin=0.15)  # 0.15 is a good margin for the reference point

    assert np.all(ref_point >= pareto_input.max(axis=0)), "ref must be >= worst point (min space)."


    print("MIN-SPACE sanity:")
    print("pareto_min min:", pareto_input.min(axis=0), " max:", pareto_input.max(axis=0), flush=True)
    print("ref_point:", ref_point, flush=True)
    print("ref >= worst? ", np.all(ref_point >= pareto_input.max(axis=0)), flush=True)
        
    hv_base = hypervolume(pareto_input).compute(ref_point)
    print(f"Base hypervolume of the Pareto front: {hv_base}", flush=True)

   

    ga_B2 = ga(function=lambda seq: fitness_function_batch(seq, featurizer, pareto_input, hv_base, ref_point, \
                                                           model, normalization_stats, front=args.front),
               algorithm_parameters={
                   'max_num_iteration': 100,
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


    # todo: potentially re-add the comparison between the two methods in some efficient way

    # ## get fitness for best sequence using old method
    # seq_list = back_AA(output_sequence) 
    # #seq_feats_list = [featurizer.featurize(seq) for seq in sequences]
    # seq_feats_list = featurizer.featurize(seq_list)
    # seq_feats_list_arr = np.asarray(seq_feats_list)
    # # seq_feats_list_arr = seq_feats_list_arr[np.newaxis, :]  # Add a new axis to make it 2D
    # seq_feats_normal_list = standard_normalize_features(seq_feats_list_arr, normalization_stats)
    # seq_arr = seq_feats_normal_list[np.newaxis, :] 
    # #seq_arr = np.array(seq_feats_normal_list)

    # # Convert all features to a tensor batch
    # seq_tensor_batch = torch.tensor(seq_arr).float()

    # # make prediction for this sequence
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     output = likelihood(model(seq_tensor_batch))
    #     mean = output.mean.numpy()
    #     var = output.stddev.numpy()

    # print(f"Predicted B2: {mean[0, 0]}, Predicted diff: {mean[0, 1]}", flush=True)

    # if args.front == 'upper':
    #     pareto_input *= -1  # transform to third quadrant for minimization
    

    # augmented_front = ehvi.front_augmentation(pareto_input, args.front)

    # # make sure that mean and var are reshaped to be (1, 1) , eg in array format
    # pred_B2_reshaped = np.array([mean[0, 0]]).reshape(-1, 1)
    # std_B2 =np.array([var[0, 0]]).reshape(-1, 1)
    # pred_diff_reshaped = np.array([mean[0, 1]]).reshape(-1, 1)
    # std_diff = np.array([var[0, 1]]).reshape(-1, 1)

    # if args.front == 'lower':
    #     ehvi_values = ehvi.ehvi_maximization(pred_B2_reshaped, std_B2, pred_diff_reshaped, std_diff, augmented_front)
    # elif args.front == 'upper':
    #     ehvi_values = ehvi.ehvi_maximization(-pred_B2_reshaped, std_B2, -pred_diff_reshaped, std_diff, augmented_front)


    # if isinstance(ehvi_values, np.ndarray):
    #     if ehvi_values.ndim > 0:
    #         ehvi_val = ehvi_values[0,0]
    #     else:
    #         ehvi_val = ehvi_values[0]
    # else:
    #     ehvi_val = ehvi_values
    # #print(f"EHVI value for sequence {args.seq_id}, candidate {args.cand_id}: {ehvi_val}", flush=True)

    # difference = np.abs(-1*output_fitness - ehvi_val)/np.abs(output_fitness)
    candidates_folder = os.path.join(args.gen_folder, f"candidates_{args.ehvi_variant}_{args.exploration_strategy}_{transform}")
    # Save the best sequence and its fitness
    save_cand_sequence(output_sequence, output_fitness, candidates_folder, args.cand_id, difference=0.0)

    
if __name__ == "__main__":
    main()
