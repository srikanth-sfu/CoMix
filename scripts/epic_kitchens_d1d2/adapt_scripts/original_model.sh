#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=baseline_run
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=96G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=0-03:00
#SBATCH -o /home/smuralid/error/tubelets/comix/ek_d1d2/original_baseline/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/comix/ek_d1d2/original_baseline/slurm-%j.err

cd $SLURM_TMPDIR
cp /project/def-mpederso/smuralid/envs/focal.zip .
unzip -qq focal.zip
module load StdEnv/2020 python/3.8.10
source focal/bin/activate
mkdir -p epic_kitchens/frames_lmdb
# D1: P08, D2: P01
cp -r /project/def-mpederso/smuralid/datasets/epic_kitchens/frames_lmdb/P0* $SLURM_TMPDIR/epic_kitchens/frames_lmdb
cp /project/def-mpederso/smuralid/datasets/epic_kitchens/epic_kitchens_bg.zip $SLURM_TMPDIR/epic_kitchens

cd epic_kitchens
unzip -qq epic_kitchens_bg.zip
cd $SLURM_TMPDIR

git clone git@github.com:srikanth-sfu/CoMix.git
cd CoMix
git checkout jester_baseline_zip
echo "------------------------------"
cp /project/def-mpederso/smuralid/datasets/epic_kitchens/d1d2.pkl .

timeout 150m python main.py --manual_seed 1 --dataset_name Epic-Kitchens --src_dataset D1 --tgt_dataset D2 \ 
--batch_size 8 --model_root /project/def-mpederso/smuralid/checkpoints/da/epic_kitchens_d1d2/original_baseline/ \
--save_in_steps 500 --log_in_steps 50 --eval_in_steps 50 --pseudo_threshold 0.7 --warmstart_models True \
 --num_iter_warmstart 4000 --num_iter_adapt 10000 --learning_rate 0.01 --learning_rate_ws 0.01 --lambda_bgm 0.01 \
 --lambda_tpl 0.01 --base_dir $SLURM_TMPDIR/epic_kitchens/ \
 --warmstart_graph /project/def-mpederso/smuralid/checkpoints/da/epic_kitchens_d1d2/original_baseline_ws/Graph-SourceOnly-Model-Best.pth \
 --warmstart_i3d /project/def-mpederso/smuralid/checkpoints/da/epic_kitchens_d1d2/original_baseline_ws/I3D-SourceOnly-Online-Model-Best.pth
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  cd $SLURM_SUBMIT_DIR
  sbatch scripts/epic_kitchens_d1d2/adapt_scripts/original_model.sh
  # scontrol requeue $SLURM_JOB_ID
else
  echo "Script completed before timeout"
  # Exit or perform any other necessary cleanup
fi
