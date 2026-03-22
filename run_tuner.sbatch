#!/bin/bash
#SBATCH --job-name=asset_prediction
#SBATCH --output=results/asset_prediction_first_run.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodelist=gnode091
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ronit.jalihal@students.iiit.ac.in

# Print job info
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"

# Change to project directory
cd ~/asset_prediction

# Activate virtual environment
source .venv/bin/activate

# Print Python and package versions
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Run the script
python3 -u src/tuner.py --n-trials 100

# Print completion time
echo "Job finished at $(date)"
