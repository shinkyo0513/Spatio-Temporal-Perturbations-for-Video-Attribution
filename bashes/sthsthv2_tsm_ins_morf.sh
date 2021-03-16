#!/bin/bash

#$ -N sthsthv2_tsm
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -o outs_sthsthv2/tsm_ins_morf_perturb_fade.txt
#$ -j y
#$ -cwd

# source /etc/profile.d/modules.sh

# source ${HOME}/.bashrc
# conda activate pytorch1.1


# python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core8_fade --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core5_fade --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core11_fade --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core14_fade --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _fade --only_test


# SthSthV2-R(2+1)D
# Insertion + MoRF

# CM_16
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method random --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method g --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method ig --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method sg --mode ins --order most_first --num_step 128 --batch_size 16 --new_size 16 --only_test
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method sg2 --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method grad_cam --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --only_test
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method grad_cam --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --only_test --extra_label _layer3
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --extra_label _core5 --only_test
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --extra_label _core8 --only_test
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --extra_label _core11 --only_test
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method eb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --only_test
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method la --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset sthsthv2 --model tsm --vis_method gbp --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16

# SCM (#superpixel=256)
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method random --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method g --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method ig --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method sg --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method sg2 --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method grad_cam --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 --extra_label _core5 --only_test
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 --extra_label _core11 --only_test
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method eb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method la --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model tsm --vis_method gbp --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 

