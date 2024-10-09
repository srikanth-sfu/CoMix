#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=pretrain_adapt
#SBATCH --account=rrg-mpederso
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=0-03:00
#SBATCH -o /home/smuralid/error/tubelets/comix-ncl_bgmix_pretrain_adapt/ucf_hmdb/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/comix-ncl_bgmix_pretrain_adapt/ucf_hmdb/slurm-%j.err

eval "$(conda shell.bash hook)"
source activate focal
cd $SLURM_TMPDIR
mkdir ucf_hmdb
cp /scratch/smuralid/datasets/ucf_hmdb_condensed.zip $SLURM_TMPDIR/ucf_hmdb
cd ucf_hmdb
unzip -qq ucf_hmdb_condensed.zip
cd $SLURM_TMPDIR

git clone git@github.com:srikanth-sfu/CoMix.git
cd CoMix
git checkout tubelet_contrast_comix_pretrain_adapt
echo "------------------------------"
timeout 170m python main.py --manual_seed 1 --auto_resume True --dataset_name UCF-HMDB \
 --src_dataset UCF --tgt_dataset HMDB --batch_size 7 \
 --model_root /scratch/smuralid/checkpoints/da/ucf_hmdb/ncl_comix_baseline_video_pretrain_adapt/ \
 --save_in_steps 500 --num_segments 0 --log_in_steps 50 --eval_in_steps 50 --pseudo_threshold 0.7 \
 --warmstart_models True --num_iter_warmstart 4000 --num_iter_adapt 10000 --learning_rate 0.01 --learning_rate_ws 0.01 \
 --lambda_bgm 0.1 --lambda_tpl 0.01 --base_dir $SLURM_TMPDIR/ucf_hmdb/ \
 --warmstart_graph /scratch/smuralid/checkpoints/da/ucf_hmdb/ncl_baseline_ws/Graph-SourceOnly-Model-Best.pth \
 --warmstart_i3d /scratch/smuralid/checkpoints/da/ucf_hmdb/ncl_baseline_ws/I3D-SourceOnly-Online-Model-Best.pth \
 --pretrain_i3d /scratch/smuralid/checkpoints/da/ucf_hmdb/ncl_comix_baseline_pretrain/I3D-SourceOnly-Online-Model-Best.pth \
 --pretrain_graph /scratch/smuralid/checkpoints/da/ucf_hmdb/ncl_baseline_ws/Graph-SourceOnly-Model-Best.pth
if [ $? -eq 124 ]; then
  echo "The script timed out after ${MAX_HOURS} hour(s). Restarting..."
  # Call the script itself again with the same configuration
  cd $SLURM_SUBMIT_DIR
  sbatch scripts/ucf_hmdb51/adapt_scripts/ncl_baseline.sh
  # scontrol requeue $SLURM_JOB_ID
else
  echo "Script completed before timeout"
  # Exit or perform any other necessary cleanup
fi
