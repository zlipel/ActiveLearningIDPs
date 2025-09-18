import pandas as pd
import argparse

def main():

    """Merge simulation outputs into a single labels CSV.

    Parameters are read from the command line to locate simulation results for
    equation-of-state and diffusivity calculations. The resulting labels are
    appended to previous iterations when provided.

    Side Effects
    ------------
    Writes a CSV file containing all collected labels for the current
    iteration.
    """


    # Define arguments and parse.
    parser = argparse.ArgumentParser(description='Generate label csv for sequences.')
    parser.add_argument("--eos_path", type=str, required=True, help="Path to eos simulation results.")
    parser.add_argument("--diff_path", type=str, required=True, help="Path to diffusivity simulation results.")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for labels.")
    parser.add_argument("--iter", type=int, required=True, help="Iteration number.")
    parser.add_argument("--prev_path", type=str, required=False, help="Output path for previous set of labels.")
    
    args = parser.parse_args()

    eos_df = pd.read_csv(args.eos_path)
    diff_df = pd.read_csv(args.diff_path)

    # extract the relevant columns from eos_df and diff_df
    # from eos_df, get columns 'density', 'density_std', 'exp_density', 'exp_density_std'
    eos_df_subset = eos_df[['density', 'density_std', 'exp_density', 'exp_density_std']]
    # from diff_df, get columns 'diff', 'diff_std', 
    diff_df_subset = diff_df[['diff', 'diff_std']]

    # merge the two dataframes on the index, keeping colukns from both
    merged_df = pd.concat([eos_df_subset, diff_df_subset], axis=1)

    # add column named generation that is the iteration number (first column)
    merged_df.insert(args.iter, 'generation', args.iter)

    if args.iter == 0:
        merged_df.to_csv(args.output_path, index=False)
    else:
        prev_labels = pd.read_csv(args.prev_path)
        df_combined = pd.concat([prev_labels, merged_df], ignore_index=True)
        df_combined.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
    