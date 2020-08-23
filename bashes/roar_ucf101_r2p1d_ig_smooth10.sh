#!/bin/bash

# CUDA_VISIBLE_DEVICES=5 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
# --testlist_idx 1 --vis_method ig --perturb_ratio 0.0 --smoothed_perturb --smooth_sigma 10 --num_epochs 20

CUDA_VISIBLE_DEVICES=3 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.1 --smoothed_perturb --smooth_sigma 10 --num_epochs 20

CUDA_VISIBLE_DEVICES=3 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.3 --smoothed_perturb --smooth_sigma 10 --num_epochs 20

CUDA_VISIBLE_DEVICES=3 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.5 --smoothed_perturb --smooth_sigma 10 --num_epochs 20

CUDA_VISIBLE_DEVICES=3 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.7 --smoothed_perturb --smooth_sigma 10 --num_epochs 20

CUDA_VISIBLE_DEVICES=3 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.9 --smoothed_perturb --smooth_sigma 10 --num_epochs 20