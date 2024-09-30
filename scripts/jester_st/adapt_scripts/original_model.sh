#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=baseline_jester
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=0-03:00
#SBATCH -o /home/smuralid/error/tubelets/comix/jester_st/original_baseline/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/comix/jester_st/original_baseline/slurm-%j.err

eval "$(conda shell.bash hook)"
source activate focal
cd $SLURM_TMPDIR
mkdir jester
# D1: P08, D2: P01
cp /scratch/smuralid/datasets/jester/jester_source.zip $SLURM_TMPDIR/jester
cp /scratch/smuralid/datasets/jester/jester_source_BG.zip $SLURM_TMPDIR/jester
cp /scratch/smuralid/datasets/jester/jester_target.zip $SLURM_TMPDIR/jester
cp /scratch/smuralid/datasets/jester/jester_target_BG.zip $SLURM_TMPDIR/jester


cd jester
unzip -qq jester_source.zip
unzip -qq jester_source_BG.zip
unzip -qq jester_target.zip
unzip -qq jester_target_BG.zip
cd $SLURM_TMPDIR


git clone git@github.com:srikanth-sfu/CoMix.git
cd CoMix
git checkout jester_baseline_zip
echo "------------------------------"
timeout 150m python main.py --manual_seed 1 --dataset_name Jester --src_dataset S --tgt_dataset T \
        --batch_size 8 --model_root /scratch/smuralid/checkpoints/da/jester_st/original_baseline/ \
        --save_in_steps 500 --log_in_steps 50 --eval_in_steps 50 --pseudo_threshold 0.7 --warmstart_models True \
        --num_iter_warmstart 4000 --num_iter_adapt 10000 --learning_rate 0.01 --learning_rate_ws 0.01 --lambda_bgm 0.1 \
        --lambda_tpl 0.1 --base_dir $SLURM_TMPDIR/jester/ \
        --warmstart_graph /scratch/smuralid/checkpoints/da/jester_st/original_baseline_ws/Graph-SourceOnly-Model-Best.pth \
        --warmstart_i3d /scratch/smuralid/checkpoints/da/jester_st/original_baseline_ws/I3D-SourceOnly-Online-Model-Best.pth
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  cd $SLURM_SUBMIT_DIR
  sbatch scripts/jester_st/adapt_scripts/original_model.sh
  # scontrol requeue $SLURM_JOB_ID
else
  echo "Script completed before timeout"
  # Exit or perform any other necessary cleanup
fi
