#!/bin/bash

#$ -N sthsthv2_tsm
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -o outs_sthsthv2/tsm_perturb_core5_cylinder_proc.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python run_all.py --dataset sthsthv2 --model tsm --only_test --vis_method perturb \
--perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 5 \
--perturb_core_shape cylinder --batch_size 4

python process_perturb_res.py --dataset sthsthv2 --model tsm --vis_method perturb \
--only_test --extra_label _core5_cylinder
