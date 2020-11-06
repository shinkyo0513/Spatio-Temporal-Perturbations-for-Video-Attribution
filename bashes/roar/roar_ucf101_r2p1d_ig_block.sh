#!/bin/bash

# CUDA_VISIBLE_DEVICES=5 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
# --testlist_idx 1 --vis_method ig --perturb_ratio 0.0 --smoothed_perturb --smooth_sigma 10 --num_epochs 20

CUDA_VISIBLE_DEVICES=6 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.1 --perturb_by_block --num_epochs 20

CUDA_VISIBLE_DEVICES=6 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.3 --perturb_by_block --num_epochs 20

CUDA_VISIBLE_DEVICES=6 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.5 --perturb_by_block --num_epochs 20

CUDA_VISIBLE_DEVICES=6 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.7 --perturb_by_block --num_epochs 20

CUDA_VISIBLE_DEVICES=6 python model_train/train_ucf101_24_model_roar.py --model r2p1d --long_range \
--testlist_idx 1 --vis_method ig --perturb_ratio 0.9 --perturb_by_block --num_epochs 20