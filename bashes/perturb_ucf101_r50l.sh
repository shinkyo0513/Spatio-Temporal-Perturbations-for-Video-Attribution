#!/bin/bash

#$ -N ucf_r50l
#$ -l rt_F=1
#$ -l h_rt=48:00:00
#$ -o outs_ucf101/r50l_perturb_core8_train_proc.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

# CUDA_VISIBLE_DEVICES=7,8,9 python run_all.py --dataset ucf101 --model r50l --only_test --vis_method perturb \
# --perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 11 --batch_size 8 \
# --master_addr 127.0.1.1 --master_port 29501 

# python run_all.py --dataset ucf101 --model r50l --only_test --vis_method perturb \
# --perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 7 --batch_size 4

# python run_all.py --dataset ucf101 --model r50l --only_test --vis_method perturb \
# --perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 3 --batch_size 4

# python run_all.py --dataset ucf101 --model r50l --only_test --vis_method perturb \
# --perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 8 --batch_size 4 

# python run_all.py --dataset ucf101 --model r50l --only_test --vis_method perturb \
# --perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 14 --batch_size 4 

python run_all.py --dataset ucf101 --model r50l --only_train --vis_method perturb \
--perturb_niter 1200 --perturb_withcore --perturb_num_keyframe 8 --batch_size 4 --perturb_core_shape ellipsoid