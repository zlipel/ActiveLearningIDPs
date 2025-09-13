import os
import argparse
import pandas as pd
import numpy as np

def select_best_sequence(input_folder, output_file, seq_id, monte_carlo=None):
    best_fitness = float('inf')  # Assuming we minimize the fitness
    best_sequence = ""

    # Loop through all output sequences and their fitness scores
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):  # Assuming each sequence is saved as a .txt file
            with open(os.path.join(input_folder, filename), 'r') as f:
                sequence = f.readline().strip()       # Assuming sequence is on the first line
                fitness = float(f.readline().strip()) # Assuming fitness is on the second line
                if monte_carlo is not None:
                    # if mc is not none, then the third line is the differenc between the two methods
                    mc_difference = float(f.readline().strip())

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_sequence = sequence

    # Save only the best sequence to the output file (without fitness)
    with open(output_file, 'w') as out_f:
        out_f.write(f"{best_sequence}\n")

    print(f"Sequence {seq_id} saved to {output_file}")
    print(f"Best sequence: {best_sequence} with fitness {best_fitness}", flush=True)
    if monte_carlo is not None:
        return -1*best_fitness, mc_difference
    else:
        return -1*best_fitness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the best sequence from a set of generated sequences.")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing the generated sequences.")
    parser.add_argument('--output_file', type=str, required=True, help="File to save the best sequence.")
    parser.add_argument('--seq_id', type=int, required=True, help="ID of the sequence")
    parser.add_argument('--ehvi_variant', type=str, default='standard', choices=['standard', 'epsilon'], help="Type of EHVI method used")
    parser.add_argument('--exploration_strategy', type=str, default='standard', choices=['standard', 'similarity_penalty', 'kriging_believer', 'constant_liar_min', 'constant_liar_mean', 'constant_liar_max'], help="Type of exploration method.")
    parser.add_argument('--transform', type=str, default='log', choices=['yeoj', 'log'], help="Transformation applied to the labels.")
    parser.add_argument("--monte_carlo", type=str, required=False, default=None, help="Whether to use Monte Carlo sampling for EHVI calculation. If provided, it will be used as a flag.")

    args = parser.parse_args()

    transform = args.transform
        
    if args.monte_carlo is not None:
        transform = f"{args.transform}_MC"
        fitness, difference = select_best_sequence(args.input_folder, args.output_file, args.seq_id, monte_carlo=args.monte_carlo)
    else:
        fitness = select_best_sequence(args.input_folder, args.output_file, args.seq_id)

    # save EHVI values to csv file that is in same parent folder as output_file

    output_folder = os.path.dirname(args.output_file)
    csv_path = os.path.join(output_folder, f'ehvi_values_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.csv')
    

    if args.monte_carlo is not None:
        new_row = pd.DataFrame({'Seq_ID': [args.seq_id], 'Best_Sequence': [fitness], 'MC_Difference': [difference]})
    else:
        new_row = pd.DataFrame({'Seq_ID': [args.seq_id], 'Best_Sequence': [fitness]})

    if args.seq_id == 1:
        # created csv file for the first sequence
        new_row.to_csv(csv_path, index=False)
    else:
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            updated_df.to_csv(csv_path, index=False)
        else:
            print(f"Error: {csv_path} does not exist. Creating a new file.")

    print(f"EHVI values saved to {os.path.join(output_folder, f'ehvi_values_{args.ehvi_variant}_{args.exploration_strategy}_{transform}.csv')}", flush=True)
