#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=baseline_run
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-03:00
#SBATCH -o /home/smuralid/error/tubelets/comix/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/comix/slurm-%j.err
source activate focal
cd $SLURM_TMPDIR
mkdir ucf_hmdb
cp /scratch/smuralid/datasets/ucf_hmdb_condensed.zip $SLURM_TMPDIR/ucf_hmdb
cd ucf_hmdb
unzip -qq ucf_hmdb_condensed.zip
cd $SLURM_TMPDIR

git clone git@github.com:srikanth-sfu/CoMix.git
cd CoMix
git checkout baseline_splitpretrain
timeout 179m python main.py --manual_seed 1 --dataset_name UCF-HMDB --src_dataset UCF --tgt_dataset HMDB --batch_size 8 --model_root /scratch/smuralid/checkpoints/da/ucf_hmdb/original_baseline/ --save_in_steps 500 --num_segments 0 --log_in_steps 50 --eval_in_steps 50 --pseudo_threshold 0.7 --warmstart_models True --num_iter_warmstart 4000 --num_iter_adapt 10000 --learning_rate 0.01 --learning_rate_ws 0.01 --lambda_bgm 0.1 --lambda_tpl 0.01 --base_dir $SLURM_TMPDIR/ucf_hmdb/ --warmstart_graph_checkpoint /scratch/smuralid/checkpoints/da/ucf_hmdb/original_baseline_ws/
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  sbatch scripts/ucf_hmdb51/original_model.sh
  # scontrol requeue $SLURM_JOB_ID
else
  echo "The script finished before timing out."
  # Exit or perform any other necessary cleanup
fi