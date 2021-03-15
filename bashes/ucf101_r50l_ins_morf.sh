#!/bin/bash

#$ -N ucf_r50l
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -o outs_ucf101/r50l_ins_morf_perturb_core8_fade.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python evaluation/main_causal_metric.py --dataset ucf101 --model r50l \
--vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
--multi_gpu --new_size 16 --extra_label _core8_fade --only_test

# python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 48 \
# --multi_gpu --num_superpixel 256 --extra_label _core8

# python evaluation/main_causal_metric.py --dataset ucf101 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core14

# python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 48 \
# --multi_gpu --num_superpixel 256 --extra_label _core14

# python evaluation/main_causal_metric.py --dataset ucf101 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core5_cylinder

# python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 48 \
# --multi_gpu --num_superpixel 256 --extra_label _core5_cylinder

# UCF101-R50L
# Insertion + MoRF

# CM
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method random --mode ins --order most_first --num_step 128 --batch_size 30
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method g --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method ig --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method sg --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method sg2 --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method grad_cam --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --extra_label _core5 --only_test
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --extra_label _core11 --only_test

# CM_16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method random --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method g --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method ig --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method sg --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method sg2 --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method grad_cam --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --extra_label _core5 --only_test
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --extra_label _core11 --only_test
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method eb --mode ins --order most_first --num_step 128 --batch_size 20 --new_size 16 --extra_label _abs2
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method la --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method gbp --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16

# SCM (#superpixel=256)
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method random --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method g --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method ig --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method sg --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method sg2 --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method grad_cam --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 --extra_label _core5 --only_test
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 --extra_label _core11 --only_test
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method eb --mode ins --order most_first --num_step 128 --parallel_size 20 --num_superpixel 256 --extra_label _abs2
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method la --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l --vis_method gbp --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256

