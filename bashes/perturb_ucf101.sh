#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,8,9 python run_all.py --dataset ucf101 --model r2p1d --vis_method perturb \
--perturb_niter 1200 --master_addr 127.0.1.1 --master_port 29501 --only_test \
--perturb_withcore --perturb_num_keyframe 8

# CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python run_all.py --dataset ucf101 --model r2p1d --vis_method perturb \
# --perturb_niter 1200

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python run_all.py --dataset ucf101 --model r50l --vis_method perturb \
# --perturb_niter 1200 --only_test

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python run_all.py --dataset ucf101 --model r50l --vis_method perturb \
# --perturb_niter 1200

# CUDA_VISIBLE_DEVICES=1,2,9 python run_all.py --dataset ucf101 --model r2p1d --vis_method perturb \
# --perturb_niter 1200 --master_addr 127.0.1.3 --master_port 29503

# CUDA_VISIBLE_DEVICES=1,2,3,4 python run_all.py --dataset ucf101 --model r2p1d --vis_method perturb \
# --perturb_niter 1200 --master_addr 127.0.1.3 --master_port 29503 --only_test \
# --perturb_withcore --perturb_num_keyframe 11

# CUDA_VISIBLE_DEVICES=2,3,4,7 python run_all.py --dataset ucf101 --model r2p1d --vis_method perturb \
# --perturb_niter 1200 --master_addr 127.0.1.3 --master_port 29503 --only_test \
# --perturb_withcore --perturb_num_keyframe 7

# CUDA_VISIBLE_DEVICES=2,3,4,7 python run_all.py --dataset ucf101 --model r2p1d --vis_method perturb \
# --perturb_niter 1200 --master_addr 127.0.1.3 --master_port 29503 --only_test \
# --perturb_withcore --perturb_num_keyframe 3

# CUDA_VISIBLE_DEVICES=2,3,4,7 python run_all.py --dataset ucf101 --model r50l --vis_method perturb \
# --perturb_niter 1200 --master_addr 127.0.1.3 --master_port 29503 --only_test \
# --perturb_withcore --perturb_num_keyframe 11

# CUDA_VISIBLE_DEVICES=1,2,3,4 python run_all.py --dataset ucf101 --model r50l --vis_method perturb \
# --perturb_niter 1200 --master_addr 127.0.1.3 --master_port 29503 --only_test \
# --perturb_withcore --perturb_num_keyframe 7

# CUDA_VISIBLE_DEVICES=2,3,4,7 python run_all.py --dataset ucf101 --model r50l --vis_method perturb \
# --perturb_niter 1200 --master_addr 127.0.1.3 --master_port 29503 --only_test \
# --perturb_withcore --perturb_num_keyframe 3