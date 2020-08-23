#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method g
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method ig
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method sg
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method sg2
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method grad_cam
# CUDA_VISIBLE_DEVICES=3 python evaluation/main_causal_metric.py --dataset epic --model r2p1d --vis_method perturb

# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method g
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method ig
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method sg
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method sg2
# CUDA_VISIBLE_DEVICES=2 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method grad_cam
CUDA_VISIBLE_DEVICES=8 python evaluation/main_causal_metric.py --dataset epic --model r50l --vis_method perturb --only_test