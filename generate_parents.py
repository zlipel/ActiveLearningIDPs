import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

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

def find_pareto_front(labels, kind = ['max', 'min'], objectives = ['diff', 'exp_density']):
    """
    Identify the non-dominated set based on D (diffusivity) and B2.
    Returns the non-dominated set data and the original indices from the `labels` DataFrame.
    """
    # Copy relevant columns and add original indices
    data = labels[objectives].copy()
    data['original_index'] = labels.index  # Store original indices
    
    # Drop rows where 'diff' is NaN
    data = data.dropna(subset=objectives).reset_index(drop=True)

    # Extract the objectives as a list of solutions for the Pareto front search
    solutions = data[objectives].values.tolist()  #  Convert to list of lists

    non_dominated_set = []
    non_dominated_indices = []

    for i in range(len(solutions)):
        is_dominated = False
        for j in range(len(solutions)):
            if i != j and dominates(solutions[j], solutions[i], kind=kind):  # Check if solution j dominates solution i
                is_dominated = True
                break
        
        if not is_dominated:
            non_dominated_set.append(solutions[i])
            non_dominated_indices.append(data['original_index'].iloc[i])  # Get original index
 
    # Create a DataFrame for the non-dominated set
    non_dominated_df = pd.DataFrame(non_dominated_set, columns=objectives)
    


    return non_dominated_df.reset_index(drop=True), non_dominated_indices

def main():

    # define arguments and parse
    parser = argparse.ArgumentParser(description='Generate parent files for the active learning iteration.')
    parser.add_argument("--features_path", type=str, required=True, help="Path to features csv.")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to labels csv.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for features.")
    parser.add_argument("--seq_file", type=str, required=True, help="Input sequence file.")
    parser.add_argument("--front", type=str, required=True, help="Upper or lower front.")
    parser.add_argument("--obj1", type=str, required=True, help="Objective 1.")
    parser.add_argument("--obj2", type=str, required=True, help="Objective 2.")
    parser.add_argument("--ehvi_variant", type=str, required=False, default='standard', choices=['standard', 'epsilon'], help="Type of EHVI method used.")
    parser.add_argument("--exploration_strategy", type=str, required=False, default='standard', choices=['standard', 'similarity_penalty', 'kriging_believer', 'constant_liar_min', 'constant_liar_mean', 'constant_liar_max'], help="Type of exploration method.")
    parser.add_argument("--transform", type=str, required=False, default='log', choices=['yeoj', 'log'], help="Transformation applied to the labels.")
    parser.add_argument("--monte_carlo", type=str, required=False, default=None, help="Whether to use Monte Carlo sampling for EHVI calculation. If provided, it will be used as a flag.")

    args = parser.parse_args()

    # Load the features and labels
    features = pd.read_csv(args.features_path)
    labels = pd.read_csv(args.labels_path)

    if args.front == 'upper':
        # Filter for the upper front
        kind = ['max', 'max']
    elif args.front == 'lower':
        # Filter for the lower front
        kind = ['min', 'min']
    else:
        raise ValueError("Invalid front type. Use 'upper' or 'lower'.")
    
    objectives = [args.obj1, args.obj2]

    # Find the Pareto front
    pareto_front, indices = find_pareto_front(labels, kind=kind, objectives=objectives)

    # Get the sequences corresponding to the Pareto front
    sequences = []
    with open(args.seq_file, 'r') as f:
        all_sequences = [line.strip() for line in f]
    for index in indices:
        sequences.append(all_sequences[index])

    # Get pareto front features and labels
    pareto_features = features.iloc[indices].reset_index(drop=True)
    pareto_labels = labels.iloc[indices].reset_index(drop=True)
    
    # Save the pareto front to the output directory
    os.makedirs(args.output_path, exist_ok=True)

    # check if we are doing front augmentation; eg if the when the features file name is split by '_', 
    #split_name = os.path.basename(args.features_path).split('_')[-1].split('.')

    if args.monte_carlo is not None:
        if 'NORM' in args.features_path:
            pareto_features.to_csv(os.path.join(args.output_path, f'features_parent_NORM_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}_MC.csv'), index=False)
            pareto_labels.to_csv(os.path.join(args.output_path, f'labels_parent_NORM_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}_MC.csv'), index=False)
            seq_file_name = os.path.join(args.output_path, f'sequences_parent_TEMP_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}_MC.txt')
        else:       
            pareto_features.to_csv(os.path.join(args.output_path, 'features_parent.csv'), index=False)
            pareto_labels.to_csv(os.path.join(args.output_path, 'labels_parent.csv'), index=False)
            seq_file_name = os.path.join(args.output_path, 'sequences_parent.txt')
    else:
        if 'NORM' in args.features_path:
            pareto_features.to_csv(os.path.join(args.output_path, f'features_parent_NORM_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}.csv'), index=False)
            pareto_labels.to_csv(os.path.join(args.output_path, f'labels_parent_NORM_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}.csv'), index=False)
            seq_file_name = os.path.join(args.output_path, f'sequences_parent_TEMP_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}.txt')
        else:       
            pareto_features.to_csv(os.path.join(args.output_path, 'features_parent.csv'), index=False)
            pareto_labels.to_csv(os.path.join(args.output_path, 'labels_parent.csv'), index=False)
            seq_file_name = os.path.join(args.output_path, 'sequences_parent.txt')


    with open(seq_file_name, 'w') as f:
        for seq in sequences:
            f.write(f"{seq}\n")
    print(f"Pareto front features, labels, and sequences saved to {args.output_path}")
if __name__ == "__main__":
    main()


