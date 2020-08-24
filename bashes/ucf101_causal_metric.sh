#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method g
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method ig
# CUDA_VISIBLE_DEVICES=1 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg
# CUDA_VISIBLE_DEVICES=9 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method sg2
# CUDA_VISIBLE_DEVICES=9 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method grad_cam
# CUDA_VISIBLE_DEVICES=9 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb
# CUDA_VISIBLE_DEVICES=9 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --extra_label _core11 --only_test
CUDA_VISIBLE_DEVICES=9 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --extra_label _core7 --only_test
CUDA_VISIBLE_DEVICES=9 python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d --vis_method perturb --extra_label _core3 --only_test

# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method g
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method ig
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method sg
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method sg2
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method grad_cam
# CUDA_VISIBLE_DEVICES=0 python evaluation/main_causal_metric.py --dataset ucf101 --model r50l --vis_method perturb