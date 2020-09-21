#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method g --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method ig --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method sg --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method sg2 --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method grad_cam --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method grad_cam --extra_label _layer3 --only_test --mode ins --num_step 128 --new_size 16
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --extra_label _core5 --only_test --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --extra_label _core11 --only_test --mode ins --num_step 128 --new_size 16

CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method g --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method ig --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method sg --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method sg2 --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method grad_cam --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method perturb --only_test --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method perturb --extra_label _core5 --only_test --mode ins --num_step 128 --new_size 16
CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method perturb --extra_label _core11 --only_test --mode ins --num_step 128 --new_size 16