#!/bin/bash

#$ -N epic_r50l
#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -o outs_epic/r50l_perturb_core8_fade_proc.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

# python run_all.py --dataset epic --model r50l --only_test --vis_method perturb \
# --perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 5 --batch_size 4

python run_all.py --dataset epic --model r50l --only_test --vis_method perturb \
--perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 8 --batch_size 4 --perturb_type fade

# python run_all.py --dataset epic --model r50l --only_test --vis_method perturb \
# --perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 14 --batch_size 4


