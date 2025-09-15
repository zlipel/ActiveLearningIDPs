import argparse


def main():
    """Collect child sequences and write them as GA candidates."""

    parser = argparse.ArgumentParser(description='Process child sequences and identify Pareto front.')
    parser.add_argument('--prev_iter_path', type=str, required=True, help='Path for previous iteration output.')
    parser.add_argument('--iter_path', type=str, required=True, help='Path for next iteration output.')
    parser.add_argument('--iter', type=int, required=True, help='Path for next iteration output.')
    parser.add_argument('--front', type=str, required=True, help='Path for previous iteration output.')
    parser.add_argument("--ehvi_variant", type=str, required=False, default='standard', choices=['standard', 'epsilon'], help="Type of EHVI method used.")
    parser.add_argument("--exploration_strategy", type=str, required=False, default='standard', choices=['standard', 'similarity_penalty', 'kriging_believer', 'constant_liar_min', 'constant_liar_mean', 'constant_liar_max'], help="Type of exploration method.")
    parser.add_argument("--transform", type=str, required=False, default='log', choices=['yeoj', 'log'], help="Transformation applied to the labels.")
    args = parser.parse_args()

    children = []
    for i in range(1, 25):
        with open(args.prev_iter_path + f'/children_{args.ehvi_variant}_{args.exploration_strategy}_{args.transform}/' + 'seq_child_' + str(i) + '.txt', 'r') as f:
            seq = [line.strip() for line in f]
            children.append(seq[0])

    with open(f"{args.iter_path}/simulation_candidates_gen{args.iter}_{args.front}.txt", "w") as f:
        for seq in children:
            f.write(f"{seq}\n")


if __name__ == "__main__":
    main()
