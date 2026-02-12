#!/bin/bash
#SBATCH --job-name=sqr
#SBATCH --time=72:00:00
#SBATCH --array=0-614
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq

# Load GPU drivers
module load julia/1.9.3

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate sqr-noversion

# --- CONFIGURATION LOGIC ---

# Define the list of TAU values
TAU_VALUES=(0.5 0.6 0.7 0.8 0.9)

# Define the number of datasets (0 to 122 = 123 datasets)
NUM_DATASETS=123

# Calculate the index for TAU based on the SLURM task ID
# Integer division: (0-122 -> 0), (123-245 -> 1), etc.
TAU_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_DATASETS))

# Get the actual TAU value from the array
TAU=${TAU_VALUES[$TAU_INDEX]}

# Calculate the Dataset ID
# Modulo: Cycles 0-122 repeatedly
DS_ID=$((SLURM_ARRAY_TASK_ID % NUM_DATASETS))

# Optional: Print info to the log for debugging
echo "Running Job ID: $SLURM_ARRAY_TASK_ID"
echo "Assigned TAU: $TAU"
echo "Assigned DS_ID: $DS_ID"

# --- EXECUTION ---

# Run the actual experiment.
python ~/SQR/sqr_sampling.py $DS_ID $TAU 10000

# Move results
DEST_DIR="/var/scratch/fht800/sqr_10ksampling_taus/"

# Loop through generated JSON files to rename and move them
for file in *.json; do
    # Check if file exists to avoid errors if the script failed to produce output
    [ -e "$file" ] || continue

    # Extract filename without extension (e.g., "results" from "results.json")
    base_name=$(basename "$file" .json)

    # Rename and move: appends _tau0.5 to the filename
    # Example: results.json -> /path/to/results_tau0.5.json
    mv "$file" "${DEST_DIR}/${base_name}_tau${TAU}.json"
done