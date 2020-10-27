#!/bin/bash

# Del
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method g --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method ig --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method sg --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method sg2 --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method grad_cam --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method grad_cam --extra_label _layer3 --only_test --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --extra_label _core5 --only_test --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method perturb --extra_label _core11 --only_test --mode del --num_step 100 --parallel_size 30

CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l --vis_method g --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l --vis_method ig --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l --vis_method sg --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l --vis_method sg2 --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l --vis_method grad_cam --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l --vis_method perturb --only_test --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l --vis_method perturb --extra_label _core5 --only_test --mode del --num_step 100 --parallel_size 30
CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l --vis_method perturb --extra_label _core11 --only_test --mode del --num_step 100 --parallel_size 30

# CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode del --num_superpixel 1000 --num_step 100 --parallel_size 30
# CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode del --num_superpixel 500 --num_step 100 --parallel_size 30
# CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode del --num_superpixel 200 --num_step 100 --parallel_size 30
# CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode del --num_superpixel 100 --num_step 100 --parallel_size 30
# CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode del --num_superpixel 50 --num_step 100 --parallel_size 30
# CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode del --num_superpixel 20 --num_step 100 --parallel_size 30
# CUDA_VISIBLE_DEVICES=5 python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d --vis_method random --mode del --num_superpixel 10 --num_step 100 --parallel_size 30