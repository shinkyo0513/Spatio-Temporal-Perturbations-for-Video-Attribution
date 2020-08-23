#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset epic --model r2p1d --num_gpu 1 --vis_method g 
# CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset epic --model r2p1d --num_gpu 1 --vis_method ig 
CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset epic --model r2p1d --num_gpu 1 --vis_method sg --master_addr 127.0.1.2 --master_port 29502
CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset epic --model r2p1d --num_gpu 1 --vis_method sg2 --master_addr 127.0.1.2 --master_port 29502
CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset epic --model r2p1d --num_gpu 1 --vis_method grad_cam --master_addr 127.0.1.2 --master_port 29502