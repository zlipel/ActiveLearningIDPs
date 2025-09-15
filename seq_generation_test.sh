#!/bin/bash
# Master script coordinating feature generation, model training and GA search
# for a full active learning iteration.
#SBATCH --job-name=GA_iterk
#SBATCH --output=master_gen_%j.out
#SBATCH --error=master_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=1G

# -------------------------------
# Function: Display Usage
# -------------------------------
usage() {
    echo "Usage: $0  --ITER ITER --MODEL MODEL --OBJ1 OBJ1 --OBJ2 OBJ2 --FRONT FRONT"
    echo "  --iter ITER    : Active learning iteration number"
    echo "  --model MODEL  : Model name (e.g., mpipi, urrymodel)"
    echo "  --obj1 OBJ1    : First objective to optimize"
    echo "  --obj2 OBJ2    : Second objective to optimize"
    echo "  --front FRONT  : Pareto front (upper or lower)"
    echo "  --exploration EXPLORE : which similary strategy to use "
    echo "  --ehvi_strategy) EHVI : which ehvi strat to use "
    echo "  --num_gen) NGEN : which ehvi strat to use "
    exit 1
}

# -------------------------------
# Argument Parsing
# -------------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --iter) ITER="$2"; shift ;;
        --obj1) OBJ1="$2" ; shift ;;
        --obj2) OBJ2="$2"; shift ;;
        --model) MODEL="$2"; shift ;; 
        --front) FRONT="$2"; shift ;;
        --ehvi_strategy) EHVI="$2"; shift ;;
        --exploration) EXPLORE="$2"; shift ;;
        --transform) TRANSFORM="$2"; shift ;;
        --num_gen) NGEN="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
    esac
    shift
done

# -------------------------------
# Validate Required Arguments
# -------------------------------
if [[ -z "$ITER" || -z "$MODEL" || -z "$OBJ1" || -z "$OBJ2" || -z "$FRONT" ]]; then
    usage
fi

module purge
module load anaconda3/2024.6
conda activate torch-chemistry

start_AL_time=$(date +%s)
echo "Starting active learning iteration $ITER for model $MODEL"

# -------------------------------
# Define Directory Structure
# -------------------------------
ITERATION_DIR="/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/GENERATIONS/iteration_$ITER"
BASE_DIR="/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/$FRONT/iteration_$ITER"
MODEL_DIR="/home/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/MODELS"
LOG_DIR="/home/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/logs/iteration_${FRONT}_$ITER"

mkdir -p "$BASE_DIR" "$MODEL_DIR" "$LOG_DIR"

# -------------------------------
# Sequence File Handling
# -------------------------------
if [[ $ITER -eq 0 ]]; then
    SEQ_SIM_FILE="/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/seq_init.txt"
    cp "$SEQ_SIM_FILE" "$ITERATION_DIR/seq_gen$ITER.txt"
    SEQ_FILE="$ITERATION_DIR/seq_gen$ITER.txt"
    echo "Using initial sequence file: $SEQ_FILE"
else
    SEQ_SIM_FILE="$ITERATION_DIR/SIMULATIONS/EOS/seq_gen$ITER.txt"
    echo "Using sequence file for iteration $ITER: $SEQ_SIM_FILE"
fi

# -------------------------------
# Generate Features
# -------------------------------
echo "Generating features for iteration $ITER..."
if [[ $ITER -eq 0 ]]; then
    python /home/zl4808/PROJECTS/MODEL_COMPARISON/generate_features.py \
        --seq_file "$SEQ_FILE" \
        --output_path "$ITERATION_DIR/features_gen$ITER.csv" \
        --model_name "$MODEL" \
        --db_path "/home/zl4808/scripts/GENDATA/databases"
else
    PREV_ITER_DIR="/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/GENERATIONS/iteration_$((ITER - 1))"
    python /home/zl4808/PROJECTS/MODEL_COMPARISON/generate_features.py \
        --seq_file "$SEQ_SIM_FILE" \
        --output_path "$ITERATION_DIR/features_gen$ITER.csv" \
        --model_name "$MODEL" \
        --iter "$ITER" \
        --db_path "/home/zl4808/scripts/GENDATA/databases" \
        --prev_path "$PREV_ITER_DIR/features_gen$((ITER - 1)).csv"
fi

# -------------------------------
# Generate Labels
# -------------------------------
echo "Generating labels for iteration $ITER..."
EOS_PATH="$ITERATION_DIR/SIMULATIONS/EOS/eos_results.csv"
DIFF_PATH="$ITERATION_DIR/SIMULATIONS/DIFF/diffusivities.csv"

if [[ $ITER -eq 0 ]]; then
    python -m al_pipeline.labels.generate_labels \
        --eos_path "/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/SIMULATIONS/EOS/eos_results.csv" \
        --diff_path "/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/SIMULATIONS/DIFF/diffusivities.csv" \
        --output_path "$ITERATION_DIR/labels_gen$ITER.csv" \
        --iter $ITER
else
    python -m al_pipeline.labels.generate_labels \
        --eos_path "$EOS_PATH" \
        --diff_path "$DIFF_PATH" \
        --output_path "$ITERATION_DIR/labels_gen$ITER.csv" \
        --iter $ITER \
        --prev_path "$PREV_ITER_DIR/labels_gen$((ITER - 1)).csv"
fi

# -------------------------------
# Train GPR Models
# -------------------------------
echo "Training GPR models for iteration $ITER..."

python /home/zl4808/PROJECTS/MODEL_COMPARISON/train_gpr_multitask.py \
    --features "$ITERATION_DIR/features_gen$ITER.csv" \
    --labels "$ITERATION_DIR/labels_gen$ITER.csv" \
    --label_column1 "$OBJ1" \
    --label_column2 "$OBJ2" \
    --epochs 1000 \
    --lr 0.1 \
    --patience 5 \
    --batch_size 32 \
    --model_save_path "$MODEL_DIR/GPR_iter${ITER}_${EHVI}_${EXPLORE}_${TRANSFORM}" \
    --exploration "$EXPLORE" \
    --ehvi_variant "$EHVI" \
    --transform "$TRANSFORM" 



# -------------------------------
# Generate Parent Sequences
# -------------------------------
echo "Generating parent sequences for iteration $ITER..."
if [[ $ITER -eq 0 ]]; then
    python -m al_pipeline.selection.generate_parents \
        --features_path "$ITERATION_DIR/features_gen${ITER}_NORM_${EHVI}_${EXPLORE}_${TRANSFORM}.csv" \
        --labels_path "$ITERATION_DIR/labels_gen${ITER}_NORM_${EHVI}_${EXPLORE}_${TRANSFORM}.csv" \
        --output_path "$BASE_DIR" \
        --seq_file "$SEQ_FILE" \
        --front "$FRONT" \
        --obj1 "$OBJ1" \
        --obj2 "$OBJ2" \
        --ehvi_variant "$EHVI" \
        --exploration_strategy "$EXPLORE" \
        --transform "$TRANSFORM"
else
    CURR_SEQ_FILE="$ITERATION_DIR/seq_simulated_gen$ITER.txt"
    PREV_SEQ_FILE="$PREV_ITER_DIR/seq_gen$((ITER - 1)).txt"
    cp "$SEQ_SIM_FILE" "$CURR_SEQ_FILE"
    SEQ_FILE="$ITERATION_DIR/seq_gen$ITER.txt"
    cat "$PREV_SEQ_FILE" "$CURR_SEQ_FILE" > "$SEQ_FILE"
    python -m al_pipeline.selection.generate_parents \
        --features_path "$ITERATION_DIR/features_gen${ITER}_NORM_${EHVI}_${EXPLORE}_${TRANSFORM}.csv" \
        --labels_path "$ITERATION_DIR/labels_gen${ITER}_NORM_${EHVI}_${EXPLORE}_${TRANSFORM}.csv" \
        --output_path "$BASE_DIR" \
        --seq_file "$SEQ_FILE" \
        --front "$FRONT" \
        --obj1 "$OBJ1" \
        --obj2 "$OBJ2" \
        --ehvi_variant "$EHVI" \
        --exploration_strategy "$EXPLORE" \
        --transform "$TRANSFORM"
fi

cp "${BASE_DIR}/sequences_parent_TEMP_${EHVI}_${EXPLORE}_${TRANSFORM}.txt" "${BASE_DIR}/sequences_parent.txt"


# -------------------------------
# Calculate Normalization Stats
# -------------------------------
echo "Calculating normalization statistics..."
python -m al_pipeline.features.calculate_normalization_stats \
    --features_file "$ITERATION_DIR/features_gen$ITER.csv" \
    --output_file "$ITERATION_DIR/normalization_stats.json"



# -------------------------------
# Generate Child Sequences
# -------------------------------
echo "Generating child sequences for iteration $ITER..."
CHILDREN_DIR="$BASE_DIR/children_${EHVI}_${EXPLORE}_${TRANSFORM}"
rm -rf "$CHILDREN_DIR" && mkdir -p "$CHILDREN_DIR"

for SEQ_ID in $(seq 1 $NGEN); do
    echo "Generating child sequence $SEQ_ID..."
    JOB_ID=$(sbatch --parsable ga_run_ehvi_test.sh $SEQ_ID $ITER $MODEL $OBJ1 $OBJ2 $FRONT $EHVI $EXPLORE $NGEN $TRANSFORM)
    
    # Poll Job Status
    while true; do
        STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | tail -n1 | awk '{print $1}')
        if [[ "$STATUS" == "COMPLETED" ]]; then
            if [[ -f "$CHILDREN_DIR/child_seq_${SEQ_ID}_done_${EXPLORE}_${TRANSFORM}.flag" ]]; then
                echo "Child sequence $SEQ_ID completed."
                break
            fi
        elif [[ "$STATUS" == "FAILED" || "$STATUS" == "CANCELLED" ]]; then
            echo "Job $JOB_ID failed. Exiting..."
            exit 1
        fi

        sleep 1
    done

    rm "$CHILDREN_DIR/child_seq_${SEQ_ID}_done_${EXPLORE}_${TRANSFORM}.flag"  # Clean up the flag file

done

end_AL_time=$(date +%s)
elapsed_AL=$((end_AL_time - start_AL_time))
echo "Finished generating all child sequences for active learning iteration $ITER in $elapsed_AL seconds."

#exit 0
# mv master_output_${SLURM_JOB_ID}.out "$LOG_DIR/master_output_${FRONT}_iter$ITER.out"
# mv master_error_${SLURM_JOB_ID}.err "$LOG_DIR/master_error_${FRONT}_iter$ITER.err"
echo "Active learning iteration $ITER completed."

#exit 0 #comment out the exit code to avoid premature termination

# -------------------------------
# Simulation Preparation
# -------------------------------
NEXT_ITER=$((ITER + 1))
NEXT_GEN_DIR="/scratch/gpfs/zl4808/PROJECTS/MODEL_COMPARISON/$MODEL/GENERATIONS/iteration_$NEXT_ITER/SIMULATIONS"
mkdir -p "$NEXT_GEN_DIR/EOS" "$NEXT_GEN_DIR/DIFF"

# generate simulation candidates list
python /home/zl4808/PROJECTS/MODEL_COMPARISON/generate_candidate_seqs.py \
    --prev_iter_path "$BASE_DIR" \
    --iter_path $NEXT_GEN_DIR \
    --iter $NEXT_ITER \
    --front $FRONT \
    --ehvi_variant "$EHVI" \
    --exploration_strategy "$EXPLORE" \
    --transform "$TRANSFORM"

cp "$NEXT_GEN_DIR/simulation_candidates_gen${NEXT_ITER}_${FRONT}.txt" "$NEXT_GEN_DIR/DIFF/simulation_candidates_gen${NEXT_ITER}_${FRONT}.txt"
cp "$NEXT_GEN_DIR/simulation_candidates_gen${NEXT_ITER}_${FRONT}.txt" "$NEXT_GEN_DIR/EOS/simulation_candidates_gen${NEXT_ITER}_${FRONT}.txt"

# -------------------------------
# Copy Logs
# -------------------------------
mv master_gen_${SLURM_JOB_ID}.out "$LOG_DIR/master_gen_${FRONT}_iter$ITER.out"
mv master_gen_${SLURM_JOB_ID}.err "$LOG_DIR/master_gen_${FRONT}_iter$ITER.err"
echo "Active learning iteration $ITER completed."