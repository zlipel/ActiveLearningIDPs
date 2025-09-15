import pandas as pd
import json
import argparse

def calculate_normalization_stats(features_file, output_file):

    """Compute normalization statistics for feature scaling.

    Parameters
    ----------
    features_file : str
        Path to the CSV file containing raw features.
    output_file : str
        Destination for the JSON file with normalization parameters.

    Side Effects
    ------------
    Writes a JSON file containing means, standard deviations and range
    information used for later feature normalization.
    """


    df_inp = pd.read_csv(features_file)

    features_df = df_inp.copy()
    # Convert counts to prevalence for amino acids (first 20 columns)
    for col in features_df.columns[:20]:
        features_df[col] = features_df[col] / features_df['length']
    
    # Convert total charges to fractions
    features_df['beads(+)'] = features_df['beads(+)'] / features_df['length']
    features_df['beads(-)'] = features_df['beads(-)'] / features_df['length']

    features_df['|net charge|'] = features_df['|net charge|']/features_df['length']

    features_df['sum lambda'] = features_df['sum lambda']/features_df['length']

    features_df['mol wt'] = features_df['mol wt']/features_df['length']

    # Specify the features that need standard normalization
    std_norm_feats = ['beads(+)', 'beads(-)', 'sum lambda', 'mol wt', 'SHD', 'SCD', '|net charge|']
    std_normal_dict = {}

    # Compute the mean and standard deviation for standard normalization
    for feat in std_norm_feats:
        mean_val = features_df[feat].mean()
        std_val = features_df[feat].std()
        std_normal_dict[feat] = (mean_val, std_val)

    # Max value for Shannon entropy normalization
    maxS = features_df['shan ent'].max()

    # Store Min-Max normalization for sequence length
    min_L = features_df['length'].min()
    max_L = features_df['length'].max()

    # Save the normalization statistics to a JSON file
    normalization_stats = {
        'std_normal_dict': std_normal_dict,
        'min_L': min_L,
        'max_L': max_L,
        'maxS': maxS
    }

    with open(output_file, 'w') as f:
        json.dump(normalization_stats, f)

    print(f"Normalization statistics saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Calculate normalization statistics and save to JSON.")
    parser.add_argument('--features_file', required=True, help="Path to the input features CSV file.")
    parser.add_argument('--output_file', required=True, help="Path to save the output normalization stats JSON file.")

    args = parser.parse_args()

    # Call the function with the provided arguments
    calculate_normalization_stats(args.features_file, args.output_file)

