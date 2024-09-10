#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=baseline_run
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --time=0-00:30
source activate focal
cd $SLURM_TMPDIR
timeout 15s test.py
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  sbatch scripts/ucf_hmdb51/original_model.sh
  # scontrol requeue $SLURM_JOB_ID
else
  echo "The script finished before timing out."
  # Exit or perform any other necessary cleanup
fi