#!/bin/bash

# UCF101-R(2+1)D
# Insertion + LeRF

# CM
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode ins --order least_first --num_step 128 --batch_size 30
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method g --mode ins --order least_first --num_step 128 --batch_size 30
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method ig --mode ins --order least_first --num_step 128 --batch_size 30
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg --mode ins --order least_first --num_step 128 --batch_size 30
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg2 --mode ins --order least_first --num_step 128 --batch_size 30
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method grad_cam --mode ins --order least_first --num_step 128 --batch_size 30
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --mode ins --order least_first --num_step 128 --batch_size 30

# CM_16
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode ins --order least_first --num_step 128 --batch_size 30 --new_size 16
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method g --mode ins --order least_first --num_step 128 --batch_size 30 --new_size 16
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method ig --mode ins --order least_first --num_step 128 --batch_size 30 --new_size 16
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg --mode ins --order least_first --num_step 128 --batch_size 30 --new_size 16
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg2 --mode ins --order least_first --num_step 128 --batch_size 30 --new_size 16
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method grad_cam --mode ins --order least_first --num_step 128 --batch_size 30 --new_size 16
CUDA_VISIBLE_DEVICES=4 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --mode ins --order least_first --num_step 128 --batch_size 30 --new_size 16

# SCM (#superpixel=256)
CUDA_VISIBLE_DEVICES=4 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode ins --order least_first --num_step 128 --parallel_size 30 --num_superpixel 256 
CUDA_VISIBLE_DEVICES=4 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r2p1d --vis_method g --mode ins --order least_first --num_step 128 --parallel_size 30 --num_superpixel 256 
CUDA_VISIBLE_DEVICES=4 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r2p1d --vis_method ig --mode ins --order least_first --num_step 128 --parallel_size 30 --num_superpixel 256 
CUDA_VISIBLE_DEVICES=4 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg --mode ins --order least_first --num_step 128 --parallel_size 30 --num_superpixel 256 
CUDA_VISIBLE_DEVICES=4 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg2 --mode ins --order least_first --num_step 128 --parallel_size 30 --num_superpixel 256 
CUDA_VISIBLE_DEVICES=4 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r2p1d --vis_method grad_cam --mode ins --order least_first --num_step 128 --parallel_size 30 --num_superpixel 256 
CUDA_VISIBLE_DEVICES=4 python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --mode ins --order least_first --num_step 128 --parallel_size 30 --num_superpixel 256 

