#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on our clusters. 
# ---------------------------------------------------------------------
#SBATCH --job-name=baseline_run
#SBATCH --account=def-mpederso
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=0-00:30
#SBATCH -o /home/smuralid/error/tubelets/comix/slurm-%j.out  # Write the log on scratch
#SBATCH -e /home/smuralid/error/tubelets/comix/slurm-%j.err
source activate focal
cp /scratch/smuralid/datasets/ucf_hmdb_condensed.zip $SLURM_TMPDIR
cd $SLURM_TMPDIR
unzip ucf_hmdb_condensed.zip

git clone git@github.com:srikanth-sfu/CoMix.git
cd CoMix
git checkout baseline_splitpretrain
python main.py --manual_seed 1 --dataset_name UCF-HMDB \
--src_dataset UCF --tgt_dataset HMDB --batch_size 4 --model_root /scratch/smuralid/checkpoints/da/ucf_hmdb/original_baseline/ --save_in_steps 500 \
--log_in_steps 50 --eval_in_steps 50 --pseudo_threshold 0.7 --warmstart_models True --num_iter_warmstart 4000 \
--num_iter_adapt 10000 --learning_rate 0.01 --learning_rate_ws 0.01 --lambda_bgm 0.1 --lambda_tpl 0.01 \
--base_dir $SLURM_TMPDIR/ucf_hmdb/ --warmstart_graph_checkpoint /scratch/smuralid/checkpoints/da/ucf_hmdb/original_baseline_ws/


# Resubmit if not all work has been done yet.
# You must define the function work_should_continue().
if work_should_continue; then
     sbatch ${BASH_SOURCE[0]}
fi

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------