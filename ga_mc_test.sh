#!/bin/bash
# Launch a batch of Monte Carlo GA evaluations for a single sequence.
# Each task generates candidate sequences, evaluates them with EHVI, and
# aggregates the best child for the active learning loop.
#SBATCH --job-name=GA_batch_run_iterk
#SBATCH --output=logs/seq_num_%j.out
#SBATCH --error=logs/seq_num_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=96                # 16 total tasks (aligned with the number of cores)
#SBATCH --cpus-per-task=1          # Each task requires 1 CPU
#SBATCH --mem-per-cpu=500MB        # Memory per CPU
#SBATCH --time=00:29:59            # Time limit

module purge 
module load anaconda3/2024.6

conda activate torch-chemistry

# Define variables for this batch
SEQ_ID=$1  # This is the current child sequence number, passed by the master script
ITER=$2  # Global iteration number, passed by the master script
MODEL=$3  # Model name, passed by the master script
OBJ1=$4  # Objective 1 to maximize/minimize, passed by the master script
OBJ2=$5  # Objective 2 to maximize/minimize, passed by the master script
FRONT=$6  # Which Pareto front we are looking at: upper or lower
EHVI=$7
EXPLORE=$8
NGEN=$9  # Number of generations, passed by the master script
TRANSFORM=${10}  # Transformation type, passed by the master script


echo "Starting batch for child sequence $SEQ_ID in global iteration $ITER with objectives $OBJ1 $OBJ2 for front $FRONT and model $MODEL..."

HOME_DIR="/home/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL"
BASE_DIR="/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/$FRONT/iteration_$ITER"
ITERATION_DIR="/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/GENERATIONS/iteration_$ITER"
CHILDREN_DIR="$BASE_DIR/children_${EHVI}_${EXPLORE}_${TRANSFORM}_MC"
CANDIDATES_DIR="$BASE_DIR/candidates_${EHVI}_${EXPLORE}_${TRANSFORM}_MC"
MODEL_PATH="/home/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/MODELS"
SEQ_FILE="$ITERATION_DIR/seq_gen${ITER}_TEMP_${EHVI}_${EXPLORE}_${TRANSFORM}_MC.txt"

LOG_DIR="$HOME_DIR/logs/iteration_${FRONT}_$ITER"

mkdir -p "$CANDIDATES_DIR"
mkdir -p "$LOG_DIR"

# Distribute 96 tasks across 12 cores

for CAND_ID in {1..96}; do
    echo "Generating candidate $CAND_ID for child sequence $SEQ_ID in iteration $ITER..."
    srun  -n 1 --exclusive python -m al_pipeline.selection.ga_iterk_mc_test \
        --gen_folder "$BASE_DIR" \
        --iter_folder "$ITERATION_DIR" \
        --iteration "$ITER" \
        --model_path "$MODEL_PATH" \
        --obj1 "$OBJ1" \
        --obj2 "$OBJ2" \
        --cand_id $CAND_ID \
        --seq_id $SEQ_ID \
        --front $FRONT \
        --verbose False \
        --ehvi_variant "$EHVI" \
        --exploration_strategy "$EXPLORE" \
        --transform "$TRANSFORM" &
done

wait

# Once all candidates are generated, select the best one
echo "Selecting the best candidate for child sequence $SEQ_ID..."
python -m al_pipeline.selection.select_best_sequence \
    --input_folder "$BASE_DIR/candidates_${EHVI}_${EXPLORE}_${TRANSFORM}_MC" \
    --output_file "$CHILDREN_DIR/seq_child_${SEQ_ID}.txt" \
    --seq_id $SEQ_ID \
    --ehvi_variant "$EHVI" \
    --exploration_strategy "$EXPLORE" \
    --transform "$TRANSFORM" \
    --monte_carlo "MC"

sleep 2

if [[ "$EXPLORE" == "kriging_believer" || "$EXPLORE" == "constant_liar_min" || "$EXPLORE" == "constant_liar_max" || "$EXPLORE" == "constant_liar_mean" ]]; then

    python -m al_pipeline.features.augment_features \
        --gen_folder "$BASE_DIR" \
        --iter_folder "$ITERATION_DIR" \
        --iteration "$ITER" \
        --seq_id "$SEQ_ID" \
        --model_path "$MODEL_PATH"\
        --obj1 "$OBJ1"\
        --obj2 "$OBJ2"\
        --ehvi_variant "$EHVI" \
        --exploration_strategy "$EXPLORE" \
        --transform "$TRANSFORM" \
        --monte_carlo "MC" 
        
        
    python -m al_pipeline.selection.generate_parents \
        --features_path "$ITERATION_DIR/features_gen${ITER}_NORM_${EHVI}_${EXPLORE}_${TRANSFORM}_MC.csv" \
        --labels_path "$ITERATION_DIR/labels_gen${ITER}_NORM_${EHVI}_${EXPLORE}_${TRANSFORM}_MC.csv" \
        --output_path "$BASE_DIR" \
        --seq_file "$SEQ_FILE" \
        --front "$FRONT" \
        --obj1 "$OBJ1" \
        --obj2 "$OBJ2" \
        --ehvi_variant "$EHVI" \
        --exploration_strategy "$EXPLORE" \
        --transform "$TRANSFORM" \
        --monte_carlo "MC" 
        
fi

echo "Generating UMAP and features file"
if [[ "$SEQ_ID" -eq "$NGEN" ]]; then
    echo "Generating UMAP and features file and we passed the check"
    python /home/zl4808/PROJECTS/MODEL_COMPARISON/gen_umap_test.py \
        --gen_folder "$BASE_DIR" \
        --iter_folder "$ITERATION_DIR" \
        --iteration "$ITER" \
        --model_path "$MODEL_PATH" \
        --obj1 "$OBJ1" \
        --obj2 "$OBJ2" \
        --ehvi_variant "$EHVI" \
        --exploration_strategy "$EXPLORE" \
        --transform "$TRANSFORM" \
        --monte_carlo "MC" 
fi

# Clean up the candidates folder after selection
rm $CANDIDATES_DIR/*.txt

touch "$CHILDREN_DIR/child_seq_${SEQ_ID}_done_${EXPLORE}_${TRANSFORM}_MC.flag"

echo "Child sequence $SEQ_ID in iteration $ITER completed successfully."

mv logs/seq_num_${SLURM_JOB_ID}.out $LOG_DIR/seq_${SEQ_ID}_${TRANSFORM}_MC.out
mv logs/seq_num_${SLURM_JOB_ID}.err $LOG_DIR/seq_${SEQ_ID}_${TRANSFORM}_MC.err
