#!/bin/bash

# CUDA_VISIBLE_DEVICES=5 python run_all.py --dataset ucf101 --model r2p1d --num_gpu 1 --vis_method g --master_addr 127.0.1.3 --master_port 29503
# CUDA_VISIBLE_DEVICES=5 python run_all.py --dataset ucf101 --model r2p1d --num_gpu 1 --vis_method ig --master_addr 127.0.1.3 --master_port 29503
# CUDA_VISIBLE_DEVICES=5 python run_all.py --dataset ucf101 --model r2p1d --num_gpu 1 --vis_method sg --master_addr 127.0.1.3 --master_port 29503
# CUDA_VISIBLE_DEVICES=5 python run_all.py --dataset ucf101 --model r2p1d --num_gpu 1 --vis_method sg2 --master_addr 127.0.1.3 --master_port 29503
# CUDA_VISIBLE_DEVICES=5 python run_all.py --dataset ucf101 --model r2p1d --num_gpu 1 --vis_method grad_cam --master_addr 127.0.1.3 --master_port 29503
# CUDA_VISIBLE_DEVICES=6 python run_all.py --dataset ucf101 --model r2p1d --num_gpu 1 --vis_method eb --master_addr 127.0.1.3 --master_port 29503 --batch_size 30
# CUDA_VISIBLE_DEVICES=6 python run_all.py --dataset ucf101 --model r2p1d --num_gpu 1 --vis_method la --master_addr 127.0.1.3 --master_port 29503 --batch_size 30
CUDA_VISIBLE_DEVICES=6 python run_all.py --dataset ucf101 --model r2p1d --num_gpu 1 --vis_method gbp --master_addr 127.0.1.3 --master_port 29503