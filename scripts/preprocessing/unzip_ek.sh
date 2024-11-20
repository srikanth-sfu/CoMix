python3 scripts/preprocessing/unzip.py --dataset ek
mkdir $SLURM_TMPDIR/epic_kitchens/frames 
cd $SLURM_TMPDIR/epic_kitchens/frames
ln -s ../frames_orig train
ln -s ../frames_orig test
cd $SLURM_TMPDIR/CoMix