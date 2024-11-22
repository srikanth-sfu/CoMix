#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=pretrain
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-03:00
#SBATCH -o /home/smuralid/error/tubelets/comix-ncl_bgmix_pretrain_moen/ek_d1d2/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/comix-ncl_bgmix_pretrain_moen/ek_d1d2/slurm-%j.err

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
git checkout tubelet_contrast_comix_pretrain_adapt_moco
# bash scripts/preprocessing/unzip_ek.sh

echo "------------------------------"
cp /project/def-mpederso/smuralid/datasets/epic_kitchens/d1d2.pkl .
CUDA_VISIBLE_DEVICES=0 timeout 155m python main.py --manual_seed 1 --auto_resume True  --dataset_name Epic-Kitchens \
 --src_dataset D1 --tgt_dataset D2 --batch_size 32 --model_root /project/def-mpederso/smuralid/checkpoints/da/ek/d1d2_ncl_comix_baseline_video_pretrain_adapt_moen/ \
 --save_in_steps 500 --num_segments 0 --log_in_steps 50 --eval_in_steps 50 --pseudo_threshold 0.7 --warmstart_models True \
 --num_iter_warmstart 1500 --num_iter_adapt 10000 --learning_rate 0.01 --learning_rate_ws 0.01 --lambda_bgm 0.1 --lambda_tpl 0.01 \
 --base_dir $SLURM_TMPDIR/epic_kitchens/ \
 --warmstart_graph /project/def-mpederso/smuralid/checkpoints/da/epic_kitchens_d1d2/original_baseline_ws/Graph-SourceOnly-Model-Best.pth \
 --warmstart_i3d /project/def-mpederso/smuralid/checkpoints/da/epic_kitchens_d1d2/original_baseline_ws/I3D-SourceOnly-Online-Model-Best.pth \
 --checkpoint_path_pretrain /project/def-mpederso/smuralid/checkpoints/da/ek/d1d2_ncl_comix_baseline_pretrain_moen/
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  cd $SLURM_SUBMIT_DIR
  sbatch scripts/ek/pretrain_scripts/ncl_baseline_d1d2.sh
  # scontrol requeue $SLURM_JOB_ID
else
  cd $SLURM_SUBMIT_DIR
  echo "Starting Adapt Script"
  sbatch scripts/ek/pretrain_scripts/ncl_baseline_d1d2.sh
  # Exit or perform any other necessary cleanup
fi
