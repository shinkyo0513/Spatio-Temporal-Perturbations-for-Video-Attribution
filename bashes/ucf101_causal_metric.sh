#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method g --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method ig --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg2 --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method grad_cam --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method grad_cam --extra_label _layer3 --only_test --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --extra_label _core5 --only_test --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --extra_label _core11 --only_test --mode ins --num_step 128 --new_size 16

# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method g --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method ig --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method sg --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method sg2 --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method grad_cam --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --extra_label _core5 --only_test --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb --extra_label _core11 --only_test --mode ins --num_step 128 --new_size 16


# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode ins --num_step 128
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode ins --num_step 128 --new_size 16

# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode del --num_step 128
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode del --num_step 128 --new_size 16

# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method random --mode ins --num_step 128
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method random --mode ins --num_step 128 --new_size 16

# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method random --mode del --num_step 128
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method random --mode del --num_step 128 --new_size 16

CUDA_VISIBLE_DEVICES=5 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode del --num_step 128 --batch_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode del --num_step 512 --batch_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method random --mode del --num_step 1024 --batch_size 30