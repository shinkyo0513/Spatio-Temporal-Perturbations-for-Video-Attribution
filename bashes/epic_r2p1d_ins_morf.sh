#!/bin/bash

# EPIC-R(2+1)D
# Insertion + MoRF

# CM
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode ins --order most_first --num_step 128 --batch_size 30
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method g --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method ig --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method sg --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method sg2 --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method grad_cam --mode ins --order most_first --num_step 128 --batch_size 30 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --only_test
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --extra_label _core5 --only_test
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --extra_label _core11 --only_test

# CM_16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method g --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method ig --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method sg --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method sg2 --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method grad_cam --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --extra_label _core5 --only_test
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16 --extra_label _core11 --only_test
CUDA_VISIBLE_DEVICES=6 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method eb --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
CUDA_VISIBLE_DEVICES=6 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method la --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16
CUDA_VISIBLE_DEVICES=6 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method gbp --mode ins --order most_first --num_step 128 --batch_size 30 --new_size 16

# SCM (#superpixel=256)
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method g --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method ig --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method sg --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method sg2 --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method grad_cam --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 --extra_label _core5 --only_test
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 --extra_label _core11 --only_test
# CUDA_VISIBLE_DEVICES=6 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method eb --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=6 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method la --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 
# CUDA_VISIBLE_DEVICES=6 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method gbp --mode ins --order most_first --num_step 128 --parallel_size 30 --num_superpixel 256 

